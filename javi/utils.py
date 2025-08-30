from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
import torch, pandas as pd, numpy as np, os
from bindsnet.learning import PostPre
import torch.nn.functional as F
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix, classification_report
from datetime import datetime
import os
import json
# import wandb
# from wandb_utils import log_evaluation_results
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d


def reset_voltajes(network, device='cpu'):
    network.layers['B'].v = torch.full(network.layers['B'].v.shape, -65, device=device)
    return network


def dividir(data, minimo):
    """
    Enhanced function to divide training data into normal subsequences.
    Improved temporal sequence handling with better validation.
    """
    print(f"Dividing data into sequences of minimum length {minimo}")

    intervals = []
    in_sequence = False
    sequence_start = None

    # Enhanced interval identification with validation
    for i in range(len(data)):
        current_label = data.loc[i, 'label']

        # Start of normal sequence
        if current_label == 0 and not in_sequence:
            sequence_start = i
            in_sequence = True

        # End of normal sequence
        elif current_label != 0 and in_sequence:
            intervals.append((sequence_start, i))
            in_sequence = False

    # Handle case where sequence goes to end of data
    if in_sequence:
        intervals.append((sequence_start, len(data)))

    print(f"Found {len(intervals)} normal sequences")

    # Create DataFrame with intervals
    intervals_df = pd.DataFrame(intervals, columns=['inicio', 'final'])

    subs = []
    total_valid_sequences = 0
    total_samples = 0

    # Enhanced sequence extraction with more flexible validation
    for i, row in intervals_df.iterrows():
        inicio_tmp = int(row['inicio'])
        final_tmp = int(row['final'])
        sequence_length = final_tmp - inicio_tmp

        if sequence_length >= minimo:
            sequence_data = data.iloc[inicio_tmp:final_tmp].reset_index(drop=True)

            # More flexible validation: allow sequences with some anomalies
            normal_count = (sequence_data['label'] == 0).sum()
            anomaly_count = (sequence_data['label'] == 1).sum()

            # Accept sequences that have at least some normal samples and not too many anomalies
            if normal_count >= max(10, sequence_length * 0.3):  # At least 30% normal or 10 samples
                subs.append(sequence_data)
                total_valid_sequences += 1
                total_samples += sequence_length
                print(f"  Sequence {i}: length {sequence_length}, normal: {normal_count}, anomalies: {anomaly_count}")
            else:
                print(f"  Sequence {i}: rejected (insufficient normal samples: {normal_count}/{sequence_length})")

    print(f"Extracted {total_valid_sequences} valid sequences with {total_samples} total samples")
    print(".2f")

    return subs


def padd(data, T):
    """
    Enhanced padding function with better temporal handling
    """
    lon = len(data)

    # If data length is already multiple of T, return as is
    if lon % T == 0:
        return data

    # Calculate the next multiple of T
    lon2 = ((lon // T) + 1) * T
    lon_adicional = lon2 - lon

    if lon_adicional > 0:
        print(f"Padding {lon_adicional} samples to reach multiple of T={T}")

        # Create padding data with replication of last valid values
        # This maintains temporal continuity better than NaN padding
        if len(data) > 0:
            # Use the last row as template for padding
            last_row = data.iloc[-1:].copy()

            # Create padding rows by replicating the last valid row
            padding_rows = []
            for i in range(lon_adicional):
                new_row = last_row.copy()
                # Mark as padded data (optional: could add a padding flag)
                padding_rows.append(new_row)

            padding_df = pd.concat(padding_rows, ignore_index=True)
            data = pd.concat([data, padding_df], ignore_index=True)
        else:
            # Fallback to NaN padding if no data exists
            padding_df = pd.DataFrame(np.nan, index=range(lon_adicional), columns=data.columns)
            data = pd.concat([data, padding_df], ignore_index=True)

    return data


def expandir(serie, n):
    # Crea gemelo de la serie:
    serie2 = np.zeros_like(serie)
    
    # Identificar los índices donde hay un 1:
    indices = np.where(serie == 1)[0]
    
    # Poner a 1 los valores en el rango [índice-n, índice+n]
    for idx in indices:
        start = max(0, idx - n)
        end = min(len(serie), idx + n + 1)
        serie2[start:end] = 1
    
    return pd.Series(serie2, index=serie.index)


#Función para convertir a spikes las entradas:
def podar(x,q1,q2,cuantiles=None):
    #Función que devuelve 1 (spike) si x está en el rango [q1,q2), y 0 en caso contrario.
    #Es parte de la codificación de los datos.
    
    s=torch.zeros_like(x)
    
    s[(x>=q1) & (x<q2)]=1
    return s


def convertir_data(data, T, cuantiles, snn_input_layer_neurons_size, is_train=False, device='cpu'):
    # Move cuantiles to GPU
    print('convertir_data')
    print(device)
    
    cuantiles = cuantiles.to(device)
    
    # Convert series to GPU tensor
    serie = torch.FloatTensor(data['value']).to(device)
    
    #Tomamos la longitud de la serie.
    long=serie.shape[0]
    
    #Los valores inferiores al mínimo del vector de cuantiles se sustituyen por ese mínimo.
    serie[serie<torch.min(cuantiles)]=torch.min(cuantiles)
    serie[serie>torch.max(cuantiles)]=torch.max(cuantiles)
    
    #Construimos el tensor con los datos codificados.
    serie2input=torch.cat([serie.unsqueeze(0)] * snn_input_layer_neurons_size, dim=0)
    
    for i in range(snn_input_layer_neurons_size):
        serie2input[i,:]=podar(serie2input[i,:],cuantiles[i],cuantiles[i+1])
    
    #Lo dividimos en función del tiempo de exposición T:
    secuencias = torch.split(serie2input,T,dim=1)
    
    if is_train:
        secuencias=secuencias[0:len(secuencias)-1]
    
    return secuencias


# Function to create a Gaussian kernel
def create_gaussian_kernel(kernel_size=5, sigma=1.0, device='cpu'):
    # Create a 1D Gaussian kernel directly on GPU
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
    gaussian = torch.exp(-x**2 / (2*sigma**2))
    gaussian = gaussian / gaussian.sum()  # Normalize
    return gaussian.view(1, 1, -1)  # Shape for 1D convolution


# Enhanced function with improved kernel options and better parameter handling
def create_kernel(kernel_type='gaussian', kernel_size=5, sigma=1.0, device='cpu'):
    # Ensure kernel_size is an integer and odd
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make it odd
        print(f"Adjusted kernel_size to {kernel_size} (must be odd)")

    # Ensure sigma is positive
    sigma = max(0.1, float(sigma))

    if kernel_type == 'gaussian':
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
        kernel = torch.exp(-x**2 / (2*sigma**2))
    elif kernel_type == 'laplacian':
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
        kernel = torch.exp(-torch.abs(x) / sigma)
    elif kernel_type == 'mexican_hat':
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
        kernel = (1 - x**2 / sigma**2) * torch.exp(-x**2 / (2*sigma**2))
    elif kernel_type == 'box':
        kernel = torch.ones(kernel_size, device=device)
    elif kernel_type == 'exponential':
        x = torch.linspace(0, kernel_size-1, kernel_size, device=device)
        kernel = torch.exp(-x / sigma)
    elif kernel_type == 'difference_of_gaussians':
        x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size, device=device)
        kernel1 = torch.exp(-x**2 / (2*sigma**2))
        kernel2 = torch.exp(-x**2 / (2*(sigma*2)**2))
        kernel = kernel1 - 0.5 * kernel2
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Available: gaussian, laplacian, mexican_hat, box, exponential, difference_of_gaussians")

    # Ensure kernel sums to positive value before normalization
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel = kernel / kernel_sum
    else:
        # Fallback to uniform kernel if sum is zero or negative
        print(f"Warning: Kernel sum is {kernel_sum}, using uniform kernel")
        kernel = torch.ones(kernel_size, device=device) / kernel_size

    return kernel.view(1, 1, -1)  # Shape for 1D convolution


def crear_red(snn_input_layer_neurons_size, decaimiento, umbral, nu1, nu2, n, T, 
              use_conv_layer=True, conv_params=None, device='cpu'):
    try:
        print("Starting crear_red function...")
        # Cast n to int at the beginning of the function to ensure all uses are integer
        n = int(n)
        snn_input_layer_neurons_size = int(snn_input_layer_neurons_size)
        T = int(T)
        print(f"Initialized parameters: n={n}, input_size={snn_input_layer_neurons_size}, T={T}")
        
        network = Network(dt=1.0, learning=True).to(device)
        print("Created network")
        
        # Create layers with integer dimensions
        source_layer = Input(n=snn_input_layer_neurons_size, traces=True).to(device)
        target_layer = LIFNodes(n=n, traces=True, thresh=umbral, tc_decay=decaimiento).to(device)
        print("Created source and target layers")
        
        network.add_layer(layer=source_layer, name="A")
        network.add_layer(layer=target_layer, name="B")
        print("Added layers A and B to network")
        
        conv_layer = None
        if use_conv_layer:
            try:
                print("Creating convolutional layer...")
                conv_layer = AdaptiveLIFNodes(
                    n=n,
                    traces=True, 
                    thresh=umbral,
                    tc_decay=decaimiento,
                    tc_trace=20.0
                ).to(device)
                network.add_layer(layer=conv_layer, name="C")
                print("Successfully created and added conv layer C")
            except Exception as e:
                print(f"Error creating conv layer: {e}")
                raise
        
        try:
            print("Creating basic connections...")
            print(f"Input layer size: {snn_input_layer_neurons_size}, Hidden layer size: {n}")

            # Create connections with improved initialization for better learning
            # Use Xavier/Glorot initialization for better gradient flow
            scale_factor = np.sqrt(2.0 / (snn_input_layer_neurons_size + n))  # Xavier initialization
            base_weight = scale_factor * torch.randn(snn_input_layer_neurons_size, n)

            print(f"Initial weight scale factor: {scale_factor:.4f}")
            print(f"Weight matrix shape: {base_weight.shape}")

            # Add small positive bias to encourage initial spiking
            base_weight = base_weight + 0.1

            # Apply moderate L2 regularization
            weight_norm = torch.norm(base_weight)
            print(f"Weight norm before regularization: {weight_norm:.4f}")

            if weight_norm > 1.0:  # Only scale down if too large
                base_weight = base_weight * (1.0 / weight_norm)
                print(f"Applied regularization, new norm: {torch.norm(base_weight):.4f}")

            forward_connection = Connection(
                source=source_layer,
                target=target_layer,
                w=base_weight.to(device),
                update_rule=PostPre,
                nu=nu1
            ).to(device)
            print("Created forward connection")
            
            network.add_connection(connection=forward_connection, source="A", target="B")
            print("Added forward connection to network")
            
            # Improved recurrent connections for better temporal dynamics
            # Start with a more balanced recurrent structure
            recurrent_weights = 0.05 * torch.randn(n, n)  # Random initialization

            # Add structured connectivity: local connections with decay
            for i in range(n):
                for j in range(max(0, i-5), min(n, i+6)):  # Local connectivity window
                    if i != j:  # Avoid self-connections
                        distance = abs(i - j)
                        weight_value = 0.02 * np.exp(-distance * 0.5)  # Exponential decay
                        recurrent_weights[i, j] = weight_value

            # Add small inhibitory bias
            recurrent_weights = recurrent_weights - 0.01

            # Apply moderate regularization
            recurrent_norm = torch.norm(recurrent_weights)
            if recurrent_norm > 2.0:  # More lenient threshold
                recurrent_weights = recurrent_weights * (2.0 / recurrent_norm)

            recurrent_connection = Connection(
                source=target_layer,
                target=target_layer,
                w=recurrent_weights.to(device),
                update_rule=PostPre,
                nu=nu2
            ).to(device)
            print("Created recurrent connection")
            
            network.add_connection(connection=recurrent_connection, source="B", target="B")
            print("Added recurrent connection to network")
        except Exception as e:
            print(f"Error creating basic connections: {e}")
            raise
        
        if use_conv_layer:
            try:
                print("Setting up convolutional layer parameters...")
                # Get kernel parameters and ensure integers where needed
                kernel_type = str(conv_params.get('kernel_type', 'gaussian'))
                kernel_size = int(conv_params.get('kernel_size', 5))
                sigma = float(conv_params.get('sigma', 1.0))
                norm_factor = float(conv_params.get('norm_factor', 0.5))
                exc_inh_balance = float(conv_params.get('exc_inh_balance', 0.0))
                
                print(f"Creating kernel with params: type={kernel_type}, size={kernel_size}, sigma={sigma}")
                
                # Create base kernel
                kernel = create_kernel(
                    kernel_type=kernel_type,
                    kernel_size=kernel_size,
                    sigma=sigma,
                    device=device
                )
                print("Created base kernel")
                
                # Extract the kernel values and ensure it's a tensor
                kernel_values = kernel[0, 0].clone()
                print(f"Kernel shape: {kernel.shape}, values shape: {kernel_values.shape}")
                
                # Create weight matrix directly
                weights = torch.zeros(n, n, device=device)
                center = kernel_size // 2
                print(f"Creating weight matrix with dimensions {n}x{n}")
                
                try:
                    print("Building weight matrix...")
                    # Create indices for the weight matrix
                    rows = torch.arange(n, device=device)
                    for offset in range(-center, center + 1):
                        cols = torch.clamp(rows + offset, 0, n - 1)
                        kernel_idx = offset + center
                        print(f"Processing offset {offset}, kernel_idx {kernel_idx}")
                        weights[rows, cols] = kernel_values[kernel_idx]
                except Exception as e:
                    print(f"Error in weight matrix creation: {e}")
                    print(f"Debug info - n: {n}, center: {center}, kernel_values len: {len(kernel_values)}")
                    raise
                
                print("Applying excitatory/inhibitory balance...")
                # Apply excitatory/inhibitory balance
                if exc_inh_balance != 0:
                    mask = (torch.rand(n, n, device=device) < 0.5 + exc_inh_balance/2).float()
                    weights = weights * (1 + 0.2 * mask - 0.1)
                
                # Add skip connections
                weights = weights + 0.1 * torch.eye(n, device=device)
                print("Added skip connections")
                
                print(f"Final weights shape: {weights.shape}")
                
                print("Creating convolutional connections...")
                # Create connections with improved normalization and learning
                # More aggressive normalization for better signal propagation
                norm_value = float(norm_factor) * float(kernel_size) * 2.0  # Increased multiplier
                print(f"Using norm value: {norm_value}")

                # Convert nu2 to float if it's a tuple
                nu2_value = float(nu2[0] if isinstance(nu2, tuple) else nu2)
                print(f"Using nu2 value: {nu2_value}")

                # Ensure weights have sufficient magnitude for learning
                weight_scale = torch.norm(weights)
                target_scale = 0.5  # Target scale for good learning

                if weight_scale < 0.1:  # If weights are too small, scale them up significantly
                    scale_factor = target_scale / max(weight_scale, 1e-6)
                    weights = weights * scale_factor
                    print(f"Scaled up conv weights by factor: {scale_factor:.3f} (was {weight_scale:.6f})")
                elif weight_scale > 2.0:  # If weights are too large, scale them down
                    scale_factor = target_scale / weight_scale
                    weights = weights * scale_factor
                    print(f"Scaled down conv weights by factor: {scale_factor:.3f} (was {weight_scale:.6f})")

                # Add small random noise to prevent symmetric weights
                noise_scale = 0.05
                weights = weights + noise_scale * torch.randn_like(weights)

                print(f"Final conv weights scale: {torch.norm(weights):.4f}")

                conv_connection = Connection(
                    source=target_layer,
                    target=conv_layer,
                    w=weights,
                    update_rule=PostPre,
                    nu=nu2_value,  # Use the float value instead of tuple
                    norm=norm_value
                ).to(device)
                
                network.add_connection(conv_connection, "B", "C")
                print("Added conv connection to network")
                
                # Enhanced feedback connection for better layer communication
                feedback_weights = 0.2 * torch.randn(n, n, device=device)  # Increased base scale

                # Add structured feedback: stronger self-connections and local connections
                feedback_weights = feedback_weights + 0.1 * torch.eye(n, device=device)  # Stronger diagonal

                # Add local connectivity pattern in feedback
                for i in range(n):
                    for j in range(max(0, i-3), min(n, i+4)):  # Smaller local window
                        if i != j:
                            feedback_weights[i, j] += 0.05

                feedback_connection = Connection(
                    source=conv_layer,
                    target=target_layer,
                    w=feedback_weights,
                    update_rule=PostPre,
                    nu=nu2_value * 0.8  # Slightly higher learning rate for feedback
                ).to(device)
                
                network.add_connection(feedback_connection, "C", "B")
                print("Added feedback connection to network")
            except Exception as e:
                print(f"Error in convolutional layer setup: {e}")
                print(f"Error type: {type(e)}")
                print(f"Error args: {e.args}")
                print(f"nu2 type: {type(nu2)}, value: {nu2}")  # Additional debugging info
                raise
        
        try:
            print("Creating monitors...")
            # Create monitors
            source_monitor = Monitor(
                obj=source_layer,
                state_vars=("s",),
                time=T,
            )
            target_monitor = Monitor(
                obj=target_layer,
                state_vars=("s", "v"),
                time=T,
            )
            print("Created source and target monitors")
            
            network.add_monitor(monitor=source_monitor, name="X")
            network.add_monitor(monitor=target_monitor, name="Y")
            print("Added basic monitors to network")
            
            conv_monitor = None
            if use_conv_layer:
                conv_monitor = Monitor(
                    obj=conv_layer,
                    state_vars=("s", "v"),
                    time=T,
                )
                network.add_monitor(monitor=conv_monitor, name="Conv_mon")
                print("Added conv monitor to network")
        except Exception as e:
            print(f"Error creating monitors: {e}")
            raise
            
        print("Successfully completed crear_red function")
        return [network, source_monitor, target_monitor, conv_monitor]
        
    except Exception as e:
        print(f"Top level error in crear_red: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error args: {e.args}")
        raise


def ejecutar_red(secuencias, network, source_monitor, target_monitor, conv_monitor, T, 
                use_conv_layer=True, conv_processing_type='conv', device='cpu', conv_params=None):
    sp0, sp1, sp_conv = [], [], []
    
    print('ejecutar_red')
    print(device)
    j = 1
    for i in secuencias:
        print(f'Ejecutando secuencia {j}')
        j += 1
        
        inputs = {'A': i.T.to(device)}
        network.run(inputs=inputs, time=T)
        
        spikes = {
            "X": source_monitor.get("s").to(device),
            "B": target_monitor.get("s").to(device)
        }
        
        if use_conv_layer and conv_monitor is not None:
            spikes["C"] = conv_monitor.get("s").to(device)
        
        b_spikes = spikes["B"].float()
        b_spikes_sum = b_spikes.sum(dim=2).transpose(0, 1)
        
        sp0.append(spikes['X'].cpu().sum(axis=2))
        sp1.append(spikes['B'].cpu().sum(axis=2))
        
        if use_conv_layer:
            # Set default conv_params if not provided
            if conv_params is None:
                conv_params = {
                    'kernel_type': 'gaussian',
                    'kernel_size': 5,
                    'sigma': 1.0
                }
            
            # Different types of post-processing for the convolutional layer
            if conv_processing_type == 'conv':
                # Standard convolution with proper parameters
                kernel = create_kernel(
                    kernel_type=conv_params.get('kernel_type', 'gaussian'),
                    kernel_size=int(conv_params.get('kernel_size', 5)),
                    sigma=float(conv_params.get('sigma', 1.0)),
                    device=device
                )
                conv_spikes = F.conv1d(b_spikes_sum, kernel, padding='same')
            elif conv_processing_type == 'direct':
                # Just use the spikes directly
                conv_spikes = spikes["C"].float().sum(dim=2).transpose(0, 1)
            elif conv_processing_type == 'weighted_sum':
                # Weighted sum of both layers
                b_spikes_sum = b_spikes.sum(dim=2).transpose(0, 1)
                c_spikes_sum = spikes["C"].float().sum(dim=2).transpose(0, 1)
                conv_spikes = 0.3 * b_spikes_sum + 0.7 * c_spikes_sum
            elif conv_processing_type == 'max':
                # Take maximum of both layers
                b_spikes_sum = b_spikes.sum(dim=2).transpose(0, 1)
                c_spikes_sum = spikes["C"].float().sum(dim=2).transpose(0, 1)
                conv_spikes = torch.maximum(b_spikes_sum, c_spikes_sum)
            else:
                raise ValueError(f"Unknown conv processing type: {conv_processing_type}")
                
            sp_conv.append(conv_spikes.cpu().squeeze())
        
        network = reset_voltajes(network, device=device)
    
    sp0 = torch.cat(sp0).cpu().detach().numpy()
    sp1 = torch.cat(sp1).cpu().detach().numpy()
    
    if use_conv_layer:
        sp_conv = torch.cat(sp_conv).cpu().detach().numpy()
    else:
        sp_conv = None
    
    return [sp0, sp1, sp_conv, network]


def guardar_resultados(spikes, spikes_conv, data_test, n, snn_input_layer_neurons_size, n_trial, date_starting_trials, dataset_name, snn_process_layer_neurons_size, trial, use_conv_layer=False):
    # Create directory structure with conv layer information
    conv_dir = "conv_true" if use_conv_layer else "conv_false"
    # Get the parent directory (project root) to ensure resultados is created at the right level
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from javi/ to project root
    base_path = os.path.join(project_root, 'resultados', conv_dir, dataset_name, f'n_{snn_process_layer_neurons_size}', date_starting_trials, f'trial_{n_trial}')
    os.makedirs(base_path, exist_ok=True)

    # Save spikes
    np.savetxt(f'{base_path}/spikes', spikes, delimiter=',')
    
    # Only save spikes_conv if it exists
    if spikes_conv is not None:
        np.savetxt(f'{base_path}/spikes_conv', spikes_conv, delimiter=',')

    # Convert and save labels - handle NA values properly
    labels = data_test['label'].replace([np.inf, -np.inf], np.nan)
    labels = labels.astype(float)
    labels = labels.to_numpy()
    np.savetxt(f'{base_path}/label', labels, delimiter=',')

    # Convert and save values - handle NA values properly 
    values = data_test['value'].replace([np.inf, -np.inf], np.nan)
    values = values.astype(float)
    values = values.to_numpy()
    np.savetxt(f'{base_path}/value', values, delimiter=',')

    # Save timestamps
    timestamps = data_test['timestamp'].replace([np.inf, -np.inf], np.nan)
    timestamps = timestamps.astype(float)
    timestamps = timestamps.to_numpy()
    np.savetxt(f'{base_path}/timestamp', timestamps, delimiter=',')

    # Create DataFrame with 1D arrays
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'value': values,
        'label': labels
    })

    # Save to CSV with same format as original
    results_df.to_csv(f'{base_path}/data_test.csv', 
                     index=False,
                     float_format='%.6f')

    # Process ground truth labels
    ground_truth_labels = data_test['label'].astype(float).to_numpy()
    ground_truth_labels = np.nan_to_num(ground_truth_labels, nan=0.0)

    # Process layer B spikes
    spikes_1d = spikes.sum(axis=1) if len(spikes.shape) > 1 else spikes
    binary_predictions_B = (spikes_1d > 0).astype(float)
    predicted_anomalies_B = np.nan_to_num(binary_predictions_B, nan=0.0)
    
    # Enhanced metrics calculation for layer B
    mse_B = mean_squared_error(ground_truth_labels, predicted_anomalies_B)
    f1_B = f1_score(ground_truth_labels, predicted_anomalies_B, zero_division=0)
    precision_B = precision_score(ground_truth_labels, predicted_anomalies_B, zero_division=0)
    recall_B = recall_score(ground_truth_labels, predicted_anomalies_B, zero_division=0)

    # Additional metrics for layer B
    try:
        auc_B = roc_auc_score(ground_truth_labels, spikes_1d)
        fpr_B, tpr_B, _ = roc_curve(ground_truth_labels, spikes_1d)
        precision_curve_B, recall_curve_B, _ = precision_recall_curve(ground_truth_labels, spikes_1d)
        auc_pr_B = auc(recall_curve_B, precision_curve_B)
    except Exception as e:
        print(f"Warning: Could not calculate AUC for layer B: {e}")
        auc_B = auc_pr_B = 0.5
        fpr_B = tpr_B = precision_curve_B = recall_curve_B = np.array([])

    # Confusion matrix for layer B
    cm_B = confusion_matrix(ground_truth_labels, predicted_anomalies_B)

    # Classification report for layer B
    report_B = classification_report(ground_truth_labels, predicted_anomalies_B, output_dict=True, zero_division=0)
    
    # Save binary predictions for layer B
    np.savetxt(f'{base_path}/spikes_1d_B', spikes_1d, delimiter=',')
    np.savetxt(f'{base_path}/binary_predictions_B', binary_predictions_B, delimiter=',')
    
    # Initialize conv layer metrics as None
    mse_C, f1_C, precision_C, recall_C = None, None, None, None
    
    # Process convolutional layer results if available
    if spikes_conv is not None:
        # Process layer C spikes in the same way as layer B
        spikes_conv_1d = spikes_conv.sum(axis=1) if len(spikes_conv.shape) > 1 else spikes_conv
        binary_predictions_C = (spikes_conv_1d > 0).astype(float)
        predicted_anomalies_C = np.nan_to_num(binary_predictions_C, nan=0.0)
        
        # Enhanced metrics calculation for layer C
        mse_C = mean_squared_error(ground_truth_labels, predicted_anomalies_C)
        f1_C = f1_score(ground_truth_labels, predicted_anomalies_C, zero_division=0)
        precision_C = precision_score(ground_truth_labels, predicted_anomalies_C, zero_division=0)
        recall_C = recall_score(ground_truth_labels, predicted_anomalies_C, zero_division=0)

        # Additional metrics for layer C
        try:
            auc_C = roc_auc_score(ground_truth_labels, spikes_conv_1d)
            fpr_C, tpr_C, _ = roc_curve(ground_truth_labels, spikes_conv_1d)
            precision_curve_C, recall_curve_C, _ = precision_recall_curve(ground_truth_labels, spikes_conv_1d)
            auc_pr_C = auc(recall_curve_C, precision_curve_C)
        except Exception as e:
            print(f"Warning: Could not calculate AUC for layer C: {e}")
            auc_C = auc_pr_C = 0.5
            fpr_C = tpr_C = precision_curve_C = recall_curve_C = np.array([])

        # Confusion matrix for layer C
        cm_C = confusion_matrix(ground_truth_labels, predicted_anomalies_C)

        # Classification report for layer C
        report_C = classification_report(ground_truth_labels, predicted_anomalies_C, output_dict=True, zero_division=0)
        
        # Save binary predictions for layer C
        np.savetxt(f'{base_path}/spikes_1d_C', spikes_conv_1d, delimiter=',')
        np.savetxt(f'{base_path}/binary_predictions_C', binary_predictions_C, delimiter=',')
        
        # Create DataFrame with convolutional layer results
        results_conv_df = pd.DataFrame({
            'timestamp': timestamps,
            'value': values,
            'label': spikes_conv_1d
        })
        
        # Save to CSV
        results_conv_df.to_csv(f'{base_path}/results_conv.csv', 
                             index=False,
                             float_format='%.6f')
        
        print("MSE capa C:", mse_C)
        print("F1 capa C:", f1_C)
        print("Precision capa C:", precision_C)
        print("Recall capa C:", recall_C)
        
        # Save metrics to files
        with open(f'{base_path}/MSE_capa_C', 'w') as f:
            f.write(f'{mse_C}\n')
        with open(f'{base_path}/F1_capa_C', 'w') as f:
            f.write(f'{f1_C}\n')
        with open(f'{base_path}/Precision_capa_C', 'w') as f:
            f.write(f'{precision_C}\n')
        with open(f'{base_path}/Recall_capa_C', 'w') as f:
            f.write(f'{recall_C}\n')
    
    # Save metrics for layer B to files
    print("MSE capa B:", mse_B)
    print("F1 capa B:", f1_B)
    print("Precision capa B:", precision_B)
    print("Recall capa B:", recall_B)
    
    with open(f'{base_path}/MSE_capa_B', 'w') as f:
        f.write(f'{mse_B}\n')
    with open(f'{base_path}/F1_capa_B', 'w') as f:
        f.write(f'{f1_B}\n')
    with open(f'{base_path}/Precision_capa_B', 'w') as f:
        f.write(f'{precision_B}\n')
    with open(f'{base_path}/Recall_capa_B', 'w') as f:
        f.write(f'{recall_B}\n')
        
    # Enhanced config JSON with comprehensive metrics
    info = {
        # Network parameters
        "nu1": trial.params['nu1'],
        "nu2": trial.params['nu2'],
        "threshold": trial.params['threshold'],
        "decay": trial.params['decay'],
        "snn_input_layer_neurons_size": snn_input_layer_neurons_size,
        "snn_process_layer_neurons_size": snn_process_layer_neurons_size,

        # Layer B metrics
        "mse_B": mse_B,
        "f1_B": f1_B,
        "precision_B": precision_B,
        "recall_B": recall_B,
        "auc_B": auc_B,
        "auc_pr_B": auc_pr_B,
        "confusion_matrix_B": cm_B.tolist(),
        "classification_report_B": report_B,

        # Layer C metrics (if available)
        "mse_C": mse_C,
        "f1_C": f1_C,
        "precision_C": precision_C,
        "recall_C": recall_C,
        "auc_C": auc_C if 'auc_C' in locals() else None,
        "auc_pr_C": auc_pr_C if 'auc_pr_C' in locals() else None,
        "confusion_matrix_C": cm_C.tolist() if 'cm_C' in locals() else None,
        "classification_report_C": report_C if 'report_C' in locals() else None,

        # Best combined metrics
        "best_f1": max(f1_B, f1_C) if f1_C is not None else f1_B,
        "best_layer": "C" if (f1_C is not None and f1_C > f1_B) else "B",

        # Dataset statistics
        "anomaly_ratio": np.mean(ground_truth_labels),
        "total_samples": len(ground_truth_labels),
        "anomaly_samples": int(np.sum(ground_truth_labels)),
        "normal_samples": int(len(ground_truth_labels) - np.sum(ground_truth_labels))
    }
    
    with open(f"{base_path}/config.json", "w") as f:
        json.dump(info, f, indent=4)
        
    # Log results to wandb if a run is active
    # if wandb.run is not None:
    #     wandb_metrics = {
    #         "mse_B": mse_B,
    #         "f1_B": f1_B,
    #         "precision_B": precision_B,
    #         "recall_B": recall_B
    #     }
    #     
    #     if mse_C is not None:
    #         wandb_metrics.update({
    #             "mse_C": mse_C,
    #             "f1_C": f1_C, 
    #             "precision_C": precision_C,
    #             "recall_C": recall_C
    #         })
    #         
    #     wandb.log(wandb_metrics)
    #     
    #     # Create and log spike comparison visualization
    #     if mse_C is not None:
    #         wandb.log({
    #             "spike_comparison": wandb.plot.line_series(
    #                 xs=list(range(len(predicted_anomalies_B))), 
    #                 ys=[predicted_anomalies_B, predicted_anomalies_C, ground_truth_labels],
    #                 keys=["Layer B", "Layer C", "Ground Truth"],
    #                 title="Spike Response Comparison"
    #             )
    #         })
    #     else:
    #         wandb.log({
    #             "spike_comparison": wandb.plot.line_series(
    #                 xs=list(range(len(predicted_anomalies_B))), 
    #                 ys=[predicted_anomalies_B, ground_truth_labels],
    #                 keys=["Layer B", "Ground Truth"],
    #                 title="Spike Response Comparison"
    #             )
    #         })
    
    # Return metrics for both layers
    return mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C


def find_optimal_threshold(spikes_1d, ground_truth_labels, base_path=None):
    from sklearn.metrics import roc_curve, roc_auc_score
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ground_truth_labels, spikes_1d)
    
    # Calculate the geometric mean of sensitivity and specificity
    gmeans = np.sqrt(tpr * (1-fpr))
    
    # Find the optimal threshold
    ix = np.argmax(gmeans)
    optimal_threshold = thresholds[ix]
    
    # Plot ROC curve (optional)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.plot(fpr[ix], tpr[ix], marker='o', color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Optimal Threshold = {optimal_threshold:.2f})')
    plt.savefig(f'{base_path}/roc_curve.png')
    
    return optimal_threshold


def evaluate_thresholds(spikes_1d, ground_truth_labels, thresholds=None, base_path=None):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    if thresholds is None:
        # Generate range of potential thresholds based on data distribution
        min_val, max_val = np.min(spikes_1d), np.max(spikes_1d)
        thresholds = np.linspace(min_val, max_val, 50)
    
    results = []
    for threshold in thresholds:
        binary_pred = (spikes_1d > threshold).astype(float)
        
        # Calculate various metrics
        precision = precision_score(ground_truth_labels, binary_pred, zero_division=0)
        recall = recall_score(ground_truth_labels, binary_pred, zero_division=0)
        f1 = f1_score(ground_truth_labels, binary_pred, zero_division=0)
        accuracy = accuracy_score(ground_truth_labels, binary_pred)
        mse = mean_squared_error(ground_truth_labels, binary_pred)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'mse': mse
        })
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save metrics to CSV
    results_df.to_csv(f'{base_path}/threshold_metrics.csv', index=False)
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    for metric in ['precision', 'recall', 'f1', 'accuracy']:
        plt.plot(results_df['threshold'], results_df[metric], label=metric)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Metrics by Threshold')
    plt.savefig(f'{base_path}/threshold_metrics.png')
    
    # Find optimal threshold based on F1 score
    optimal_idx = results_df['f1'].idxmax()
    optimal_threshold = results_df.loc[optimal_idx, 'threshold']
    
    return optimal_threshold, results_df


def distribution_based_threshold(spikes_1d, ground_truth_labels, base_path=None):
    # Separate spike values for normal vs anomalous points
    normal_spikes = spikes_1d[ground_truth_labels == 0]
    anomaly_spikes = spikes_1d[ground_truth_labels == 1]
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(normal_spikes, bins=50, alpha=0.5, label='Normal')
    plt.hist(anomaly_spikes, bins=50, alpha=0.5, label='Anomaly')
    plt.xlabel('Spike Count')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Spike Counts')
    plt.savefig(f'{base_path}/spike_distribution.png')
    
    # Calculate statistics
    normal_mean, normal_std = np.mean(normal_spikes), np.std(normal_spikes)
    anomaly_mean, anomaly_std = np.mean(anomaly_spikes), np.std(anomaly_spikes)
    
    # Find threshold at midpoint or based on standard deviation
    if len(anomaly_spikes) > 0:
        threshold = (normal_mean + anomaly_mean) / 2
    else:
        # If no anomalies in training, use standard deviation approach
        threshold = normal_mean + 3 * normal_std
    
    return threshold