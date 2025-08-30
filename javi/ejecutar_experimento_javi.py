import torch, pandas as pd, numpy as np, os
# from bindsnet.network import Network
# from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
# from bindsnet.network.topology import Connection
# from bindsnet.network.monitors import Monitor
# from bindsnet.analysis.plotting import plot_spikes, plot_voltages
# from bindsnet.learning import PostPre
# import torch.nn.functional as F

import argparse
import json
import optuna
# import wandb
# from wandb_utils import *

import numpy as np

from utils import *
date_starting_trials = datetime.now().strftime('%Y_%m_%d-%H_%M')  # Format includes year, month, day, hour and minute

# # Add this near the top of the file after imports
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

def experiment(nu1, nu2, a, r, n, threshold, decay, T, expansion, path, n_trial, 
               conv_params=None, conv_processing_type='weighted_sum', trial=None):

    #Lectura de datos:
    #Esperamos que estos datos tengan las columnas 'label' y 'value'.

    # Enhanced data loading and preprocessing
    print("Loading and preprocessing data...")

    data = pd.read_csv(path, na_values=['NA'])

    # Ensure correct data types
    data['value'] = data['value'].astype('float64')
    data['label'] = data['label'].astype('Int64')

    # Handle missing labels by setting to 0
    data.loc[data['label'].isna(), 'label'] = 0

    # Remove any remaining NaN values in value column using interpolation
    if data['value'].isna().any():
        print(f"Found {data['value'].isna().sum()} NaN values in value column, interpolating...")
        data['value'] = data['value'].interpolate(method='linear', limit_direction='both')

    split = len(data) // 2

    data_train = data[:split].copy()
    data_test = data[split:].copy()

    # Reset indices
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)

    # Enhanced preprocessing for training data
    # Apply robust z-score normalization
    train_values = data_train['value'][data_train['label'] != 1]

    # Use more robust statistics with fallback handling
    train_median = train_values.median()
    train_mad = (train_values - train_median).abs().median()  # Median Absolute Deviation

    # Handle edge case where MAD is too small
    if train_mad <= 1e-6:  # Very small MAD indicates data is too concentrated
        print(f"Warning: MAD is too small ({train_mad:.6f}), using standard deviation as fallback")        # Use standard deviation as fallback
        train_std = train_values.std()
        if train_std > 0:
            data_train['value_normalized'] = (data_train['value'] - train_median) / train_std
            data_test['value_normalized'] = (data_test['value'] - train_median) / train_std
        else:
            # Last resort: simple centering
            data_train['value_normalized'] = data_train['value'] - train_median
            data_test['value_normalized'] = data_test['value'] - train_median
    else:
        # Use robust z-score with MAD
        mad_scale = train_mad * 1.4826  # Scale MAD to be consistent with standard deviation
        data_train['value_normalized'] = (data_train['value'] - train_median) / mad_scale
        data_test['value_normalized'] = (data_test['value'] - train_median) / mad_scale

    # Additional preprocessing: ensure values are in reasonable range
    data_train['value_normalized'] = data_train['value_normalized'].clip(-5, 5)  # Clip extreme outliers
    data_test['value_normalized'] = data_test['value_normalized'].clip(-5, 5)

    # Expand labels for training
    print(f"Before label expansion: {data_train['label'].sum()} anomalies out of {len(data_train)} samples")
    data_train['label'] = expandir(data_train['label'], expansion)
    print(f"After label expansion: {data_train['label'].sum()} anomalies out of {len(data_train)} samples")
    print(".2f")

    # Use normalized values for quantile calculation
    normal_values = data_train['value_normalized'][data_train['label'] != 1]

    if len(normal_values) == 0:
        print("Warning: No normal values found in training data")
        normal_values = data_train['value_normalized']

    # Enhanced quantile calculation for better anomaly detection
    # Use more extreme percentiles to capture the full range
    q_low = np.percentile(normal_values, 1)   # 1st percentile (more extreme)
    q_high = np.percentile(normal_values, 99) # 99th percentile (more extreme)
    q_range = q_high - q_low

    # Extend range more aggressively for anomaly coverage
    extended_min = q_low - 0.5 * q_range  # 50% extension instead of 20%
    extended_max = q_high + 0.5 * q_range

    # Calculate number of quantiles based on data characteristics
    data_std = normal_values.std()
    if data_std > 0:
        # Adaptive quantization based on data variability
        n_quantiles = max(20, min(150, int(q_range / (data_std * r * 0.5))))
    else:
        n_quantiles = max(20, min(150, int((extended_max - extended_min) / (q_range * r))))

    print(f"Data std: {data_std:.4f}, Q1: {q_low:.4f}, Q99: {q_high:.4f}")
    print(f"Extended range: [{extended_min:.4f}, {extended_max:.4f}]")

    cuantiles = torch.FloatTensor(np.linspace(extended_min, extended_max, n_quantiles))

    print(f"Created {len(cuantiles)-1} quantization bins")
    print(f"Quantile range: [{extended_min:.3f}, {extended_max:.3f}]")
    print(f"Data median: {train_median:.3f}, MAD: {train_mad:.3f}")
    print(f"Resolution parameter r: {r}")

    #Ahora, establecemos el valor de snn_input_layer_neurons_size, que será el número de neuronas de la capa de entrada:
    snn_input_layer_neurons_size=len(cuantiles)-1

    #Crea la red.
    network, source_monitor, target_monitor, conv_monitor = crear_red(
        snn_input_layer_neurons_size, decay, threshold, nu1, nu2, n, T, 
        use_conv_layer=use_conv_layer, conv_params=conv_params, device=device
    )

    # Use normalized values for sequence processing
    data_train_for_sequences = data_train.copy()
    data_train_for_sequences['value'] = data_train_for_sequences['value_normalized']

    data_test_for_sequences = data_test.copy()
    data_test_for_sequences['value'] = data_test_for_sequences['value_normalized']

    #Dividimos el train en secuencias:
    data_train=dividir(data_train_for_sequences,T)

    #Paddeamos el test:
    data_test=padd(data_test_for_sequences,T)

    #En este punto, entrenamos para cada secuencia consecutiva del train:

    #Para cada secuencia del train, tenemos que pasarla y entrenar la red:
    network.learning=True

    for s in data_train:
        secuencias2train=convertir_data(s,T,cuantiles,snn_input_layer_neurons_size,is_train=True,device=device)
        print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
        spikes_input,spikes,spikes_conv,network=ejecutar_red(
            secuencias2train, network, source_monitor, target_monitor, conv_monitor, T,
            use_conv_layer=use_conv_layer, 
            conv_processing_type=conv_processing_type, 
            device=device
        )
        #Reseteamos los voltajes:
        network=reset_voltajes(network)

    #Ahora, el test:
    network.learning=False
    secuencias2test=convertir_data(data_test,T,cuantiles,snn_input_layer_neurons_size,is_train=False,device=device)

    print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
    spikes_input,spikes,spikes_conv,network=ejecutar_red(
        secuencias2test, network, source_monitor, target_monitor, conv_monitor, T,
        use_conv_layer=use_conv_layer, 
        conv_processing_type=conv_processing_type, 
        device=device
    )

    mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C = guardar_resultados(
        spikes, spikes_conv, data_test, n, snn_input_layer_neurons_size, n_trial,
        date_starting_trials, dataset_name, snn_process_layer_neurons_size, trial=trial, use_conv_layer=use_conv_layer
    )
    
    
    # Choose which F1 score to return based on whether we're using conv layer
    if use_conv_layer and f1_C is not None:
        # Return the better F1 score between B and C
        best_f1 = max(f1_B if f1_B is not None else 0, f1_C if f1_C is not None else 0)
    else:
        # Return layer B metrics if not using conv layer or if conv layer failed
        best_f1 = f1_B if f1_B is not None else 0
    
    return mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, best_f1

def objective(trial):

    print(f"Running trial: {trial.number + 1}")
    # More aggressive parameter ranges for better learning
    config = {
        # STDP parameters - wider range for better plasticity
        'nu1': trial.suggest_float('nu1', -1.0, 1.0),  # Doubled range
        'nu2': trial.suggest_float('nu2', -1.0, 1.0),  # Doubled range

        # Threshold - lower values for more spiking activity
        'threshold': trial.suggest_float('threshold', -70, -45),  # Lower minimum, higher maximum

        # Decay - focus on biologically plausible ranges
        'decay': trial.suggest_float('decay', 50, 200),  # Extended range
        # "nu1": 0.001657625626991273,
        # "nu2": -0.3546294590908031,
        # "threshold": -62.416319068986716,
        # "decay": 145.29873304569423
    }
    
    # Only add convolutional parameters if using a conv layer
    conv_params = None
    conv_processing_type = 'direct'
    
    if use_conv_layer:
        # Aggressive convolutional parameters for better anomaly detection
        kernel_types = ['gaussian', 'laplacian', 'mexican_hat', 'box', 'exponential', 'difference_of_gaussians']
        config['kernel_type'] = trial.suggest_categorical('kernel_type', kernel_types)

        # Smaller kernels for local pattern detection
        config['kernel_size'] = trial.suggest_int('kernel_size', 3, 7, step=2)  # Focus on smaller, more responsive kernels

        # More aggressive sigma range for better feature detection
        config['sigma'] = trial.suggest_float('sigma', 0.1, 2.0)  # Narrower range, no log scale

        # Higher norm_factor for stronger convolutional signals
        config['norm_factor'] = trial.suggest_float('norm_factor', 0.1, 5.0)  # Higher maximum

        # More balanced excitatory/inhibitory range
        config['exc_inh_balance'] = trial.suggest_float('exc_inh_balance', -0.2, 0.2)  # Narrower, more balanced range
        
        # Create conv_params from config
        conv_params = {
            'kernel_type': config['kernel_type'],
            'kernel_size': config['kernel_size'],
            'sigma': config['sigma'],
            'norm_factor': config['norm_factor'],
            'exc_inh_balance': config['exc_inh_balance']
        }
        
        # Only optimize the way convolutional output is processed
        processing_types = ['direct', 'weighted_sum', 'max']  # Remove 'conv' from options
        conv_processing_type = trial.suggest_categorical('conv_processing_type', processing_types)
        config['conv_processing_type'] = conv_processing_type
    
    print(f"config: {config}")

    #Establecemos valores para los parámetros que nos interesan:
    # nu1_pre=0.1 #Actualización de pesos presinápticos en la capa A. Valores positivos penalizan y negativos excitan.
    # nu1_post=-0.1 #Actualización de pesos postsinápticos en la capa A. Valores postivos excitan y negativos penalizan.

    # nu2_pre=0.1 #Actualización de pesos presinápticos en la capa B. Valores positivos penalizan y negativos excitan.
    # nu2_post=-0.1 #Actualización de pesos postsinápticos en la capa B. Valores postivos excitan y negativos penalizan.

    #Parámetros que definen la amplitud del rango de cuantiles.
    #La idea es que el valor mínimo para la codificación sea inferior al mínimo de los datos de entrenamiento, por un margen. El valor máximo debe ser también  mayor que el máximo de los datos por un margen.
    #Para ello, nos inventamos la variable a, que será la proporción del rango de datos de entrenamiento que inflamos por encima y por debajo:
    a=0.1
    #La resolución, r, indica cuán pequeños tomamos los rangos al codificar:
    r=0.05

    #Número de neuronas en la capa B.

    #Umbral de disparo de las neuronas LIF:
    # threshold=-52

    # #Decaimiento, en tiempo, de las neuronas LIF:
    # decay=100

    T = 250 #Tiempo de exposición. Puede influir por la parte del entrenamiento, en la inferencia no porque los voltajes se conservan.
    #Usar el máximo de T para evitar problemas con los periodos de datos.
    expansion=100
    
    nu1=(config['nu1'],config['nu1'])
    nu2=(config['nu2'],config['nu2'])
    try:
        # Run the experiment with all parameters
        mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, best_f1 = experiment(
            nu1, nu2, a, r, snn_process_layer_neurons_size, 
            config['threshold'], config['decay'], T, expansion, path, 
            trial.number + 1, conv_params=conv_params, 
            conv_processing_type=conv_processing_type, trial=trial
        )
        
        # Make sure we have a valid F1 score
        if best_f1 is not None and not np.isnan(best_f1) and not np.isinf(best_f1):
            return 1.0 - best_f1  # Convert to minimization problem (1 - F1)
        else:
            print("Warning: Invalid F1 score. Returning infinity.")
            return float('inf')
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')

if __name__ == "__main__":
    start_time = datetime.now()  # Add this line to track start time
    
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros con Optuna.')
    # Get the parent directory (project root) to ensure dataset path is correct
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from javi/ to project root
    # path = os.path.join(project_root, 'Nuevos datasets', 'iops', 'preliminar', 'train_procesado_javi', '1c35dbf57f55f5e4_filled.csv')
    path = os.path.join(project_root, 'Nuevos datasets', 'Callt2', 'preliminar', 'train_label_filled.csv')
    # dataset_name = 'iops'  # Set directly instead of parsing from path
    dataset_name = 'callt2'  # Set directly instead of parsing from path
    snn_process_layer_neurons_size=400
    use_conv_layer=True
    # path='Nuevos datasets/Callt2/preliminar\\train_label_filled.csv'
    # parser.add_argument('-d', '--data_path', type=str, default='Nuevos datasets\\Callt2\\preliminar\\train_label_filled.csv', help='Ruta al archivo de datos CSV')
    parser.add_argument('-n', '--n_trials', type=int, default=100, help='Número de trials para Optuna')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to use (cpu/gpu)')
    args = parser.parse_args()


    # Enhanced device handling with better GPU detection
    if args.device == "gpu" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        if args.device == "gpu":
            print("Warning: GPU requested but not available, falling back to CPU")
        print("Using device: CPU")

    # Set device for PyTorch
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    # Enhanced Optuna study with better convergence monitoring
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible sampling
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)  # Early pruning
    )

    # Skip convergence monitoring for compatibility with current Optuna version
    print("Note: Using basic Optuna configuration (convergence monitoring disabled for compatibility)")
    convergence_callback = None

    print(f"Starting optimization with {args.n_trials} trials...")
    print(f"Study direction: minimize (1 - F1 score)")
    print(f"Early pruning enabled with MedianPruner")

    try:
        # Prepare callbacks list
        callbacks = []
        if convergence_callback is not None:
            callbacks.append(convergence_callback)

        study.optimize(
            objective,
            n_trials=args.n_trials,
            callbacks=callbacks if callbacks else None,
            timeout=3600 * 4  # 4 hour timeout
        )

        end_time = datetime.now()  # Add this line to track end time
        duration = end_time - start_time  # Calculate duration

        print('\n' + '='*50)
        print('OPTIMIZATION RESULTS')
        print('='*50)
        print(f'Best configuration found (Trial {study.best_trial.number}):')
        print(f'F1 Score: {1.0 - study.best_value:.4f}')
        print(f'Objective value (1-F1): {study.best_value:.6f}')
        print(f'Best parameters:')
        for key, value in study.best_params.items():
            print(f'  {key}: {value}')
        print(f'\nOptimization Statistics:')
        print(f'  Total trials: {len(study.trials)}')

        # Handle trial state checking with fallback for compatibility
        try:
            completed_count = len([t for t in study.trials if t.state == optuna.TrialState.COMPLETE])
            pruned_count = len([t for t in study.trials if t.state == optuna.TrialState.PRUNED])
            failed_count = len([t for t in study.trials if t.state == optuna.TrialState.FAIL])
        except AttributeError:
            # Fallback for older Optuna versions
            completed_count = len([t for t in study.trials if hasattr(t, 'state') and str(t.state) == 'COMPLETE'])
            pruned_count = len([t for t in study.trials if hasattr(t, 'state') and str(t.state) == 'PRUNED'])
            failed_count = len([t for t in study.trials if hasattr(t, 'state') and str(t.state) == 'FAIL'])

        print(f'  Completed trials: {completed_count}')
        print(f'  Pruned trials: {pruned_count}')
        print(f'  Failed trials: {failed_count}')
        print(f'Duración total: {duration}')

        # Guardar la mejor configuración
        conv_dir = "conv_true" if use_conv_layer else "conv_false"
        # Get the parent directory (project root) to ensure resultados is created at the right level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level from javi/ to project root
        base_path = os.path.join(project_root, 'resultados', conv_dir, dataset_name, f'n_{snn_process_layer_neurons_size}', date_starting_trials)
        os.makedirs(base_path, exist_ok=True)
        best_trial_number = study.best_trial.number
        results = {
            "best_params": study.best_params,
            "best_trial": best_trial_number+1,
            "snn_process_layer_neurons_size": snn_process_layer_neurons_size,
            "device": str(device),
            "use_conv_layer": use_conv_layer,
            "best_f1": 1.0 - study.best_value if not np.isinf(study.best_value) else None,
            "amoumt_of_trials": args.n_trials,
            "duration_seconds": duration.total_seconds(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat()
        }
        with open(f"{base_path}/best_config.json", "w") as f:
            json.dump(results, f, indent=4)
            
        # if parent_wandb_run:
        #     parent_wandb_run.log({
        #         "study/best_mse_B": study.best_value,
        #         "study/best_trial": best_trial_number+1,
        #         "study/duration_seconds": duration.total_seconds(),
        #         **{f"study/best_{k}": v for k, v in study.best_params.items()}
        #     })
    finally:
        # Finish the parent wandb run
        # if parent_wandb_run:
        #     finish_wandb_run()
        pass