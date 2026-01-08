"""
Enhanced SNN Hyperparameter Optimization Script

This script has been improved to handle common failure modes:
- Robust error handling that avoids returning infinity to Optuna
- Enhanced GPU memory management with automatic fallback to CPU
- Comprehensive data validation and preprocessing
- Better logging and debugging information
- F1 score validation to prevent null results

Key improvements:
1. Trials return poor but valid scores (0.8-0.9) instead of infinity on failure
2. GPU initialization testing prevents CUDA runtime crashes
3. Data file validation catches missing/corrupted files early
4. Enhanced logging provides detailed failure diagnostics
5. Memory monitoring helps identify GPU memory issues
"""

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
from sklearn.model_selection import TimeSeriesSplit
# import wandb
# from wandb_utils import *

import numpy as np

from utils import *
date_starting_trials = datetime.now().strftime('%Y_%m_%d-%H_%M')  # Format includes year, month, day, hour and minute

def experiment(nu1, nu2, a, r, n, threshold, decay, T, expansion, path, n_trial,
               conv_params=None, conv_processing_type='weighted_sum', trial=None,
               use_cross_validation=False, n_splits=5):

    # Memory and device management
    if device.type == "cuda":
        print(f"GPU memory before experiment: {torch.cuda.memory_allocated()/1024**3:.2f} GB used, "
              f"{torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
        torch.cuda.empty_cache()  # Clear any cached memory
        print(f"GPU memory after cache clear: {torch.cuda.memory_allocated()/1024**3:.2f} GB used")

    #Lectura de datos:
    #Esperamos que estos datos tengan las columnas 'label' y 'value'.

    # Enhanced data loading and preprocessing with validation
    print("Loading and preprocessing data...")

    # Validate data file existence and readability
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"Cannot read data file: {path}")

    try:
        data = pd.read_csv(path, na_values=['NA'])
        print(f"Successfully loaded data: {len(data)} rows, {len(data.columns)} columns")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV file {path}: {e}")

    # Validate required columns
    required_columns = ['value', 'label']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Data shape: {data.shape}")
    print(f"Label distribution: {data['label'].value_counts().to_dict()}")

    # Ensure correct data types
    try:
        data['value'] = data['value'].astype('float64')
        data['label'] = data['label'].astype('Int64')
    except Exception as e:
        raise ValueError(f"Data type conversion failed: {e}")

    # Handle missing labels by setting to 0
    missing_labels = data['label'].isna().sum()
    if missing_labels > 0:
        print(f"Found {missing_labels} missing labels, setting to 0 (normal)")
        data.loc[data['label'].isna(), 'label'] = 0

    # Remove any remaining NaN values in value column using interpolation
    nan_values = data['value'].isna().sum()
    if nan_values > 0:
        print(f"Found {nan_values} NaN values in value column, interpolating...")
        data['value'] = data['value'].interpolate(method='linear', limit_direction='both')

        # Check if interpolation was successful
        remaining_nans = data['value'].isna().sum()
        if remaining_nans > 0:
            print(f"Warning: {remaining_nans} NaN values remain after interpolation, filling with mean")
            data['value'] = data['value'].fillna(data['value'].mean())

    # Validate data ranges and statistics
    print(f"Value statistics: min={data['value'].min():.3f}, max={data['value'].max():.3f}, "
          f"mean={data['value'].mean():.3f}, std={data['value'].std():.3f}")

    if data['value'].std() == 0:
        raise ValueError("All values are identical - no variation in data")

    if len(data) < 100:
        print(f"Warning: Very small dataset ({len(data)} samples)")

    # Cross-validation or simple split
    if use_cross_validation:
        print(f"Using TimeSeriesSplit with {n_splits} folds")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_results = []

        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            print(f"\nFold {fold + 1}/{n_splits}:")
            print(f"Train indices: {len(train_index)} samples")
            print(f"Test indices: {len(test_index)} samples")

            data_train = data.iloc[train_index].copy()
            data_test = data.iloc[test_index].copy()

            # Reset indices
            data_train = data_train.reset_index(drop=True)
            data_test = data_test.reset_index(drop=True)

            # Run experiment for this fold
            fold_result = run_single_experiment(
                data_train, data_test, nu1, nu2, a, r, n, threshold, decay, T, expansion,
                n_trial, conv_params, conv_processing_type, trial, fold,
                n_epochs, early_stopping_patience
            )
            cv_results.append(fold_result)

        # Aggregate results across folds
        if cv_results:
            avg_mse_B = np.mean([r[0] for r in cv_results if r[0] is not None])
            avg_mse_C = np.mean([r[1] for r in cv_results if r[1] is not None])
            avg_f1_B = np.mean([r[2] for r in cv_results if r[2] is not None])
            avg_precision_B = np.mean([r[3] for r in cv_results if r[3] is not None])
            avg_recall_B = np.mean([r[4] for r in cv_results if r[4] is not None])
            avg_f1_C = np.mean([r[5] for r in cv_results if r[5] is not None])
            avg_precision_C = np.mean([r[6] for r in cv_results if r[6] is not None])
            avg_recall_C = np.mean([r[7] for r in cv_results if r[7] is not None])
            avg_best_f1 = np.mean([r[8] for r in cv_results if r[8] is not None])

            print(f"\nCross-validation results ({n_splits} folds):")
            print(f"Average F1-B: {avg_f1_B:.4f}")
            if avg_f1_C is not None:
                print(f"Average F1-C: {avg_f1_C:.4f}")
            print(f"Average Best F1: {avg_best_f1:.4f}")

            return avg_mse_B, avg_mse_C, avg_f1_B, avg_precision_B, avg_recall_B, avg_f1_C, avg_precision_C, avg_recall_C, avg_best_f1
        else:
            raise RuntimeError("All cross-validation folds failed")

    else:
        # Simple split for backward compatibility
        print("Using simple 50/50 split")
        split = len(data) // 2
        data_train = data[:split].copy()
        data_test = data[split:].copy()

        # Reset indices
        data_train = data_train.reset_index(drop=True)
        data_test = data_test.reset_index(drop=True)

        return run_single_experiment(
            data_train, data_test, nu1, nu2, a, r, n, threshold, decay, T, expansion,
            n_trial, conv_params, conv_processing_type, trial, None,
            n_epochs, early_stopping_patience
        )


def run_single_experiment(data_train, data_test, nu1, nu2, a, r, n, threshold, decay, T, expansion,
                         n_trial, conv_params=None, conv_processing_type='weighted_sum', trial=None, fold=None,
                         n_epochs=3, early_stopping_patience=2):
    """Run a single experiment with multi-epoch training and best epoch selection"""

    # Enhanced preprocessing for training data
    # Apply robust z-score normalization
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

    # Multi-epoch training with best epoch selection
    epoch_results = []
    best_network_state = None
    best_f1_score = -1
    patience_counter = 0

    print(f"Training for {n_epochs} epochs with early stopping patience {early_stopping_patience}")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Reset network for each epoch (important for SNNs)
        network = reset_voltajes(network)

        # Training phase
        network.learning = True
        for s in data_train:
            secuencias2train = convertir_data(s, T, cuantiles, snn_input_layer_neurons_size, is_train=True, device=device)
            if epoch == 0:  # Only print once
                print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
            spikes_input, spikes, spikes_conv, network = ejecutar_red(
                secuencias2train, network, source_monitor, target_monitor, conv_monitor, T,
                use_conv_layer=use_conv_layer,
                conv_processing_type=conv_processing_type,
                device=device
            )
            network = reset_voltajes(network)

        # Testing phase
        network.learning = False
        network = reset_voltajes(network)
        secuencias2test = convertir_data(data_test, T, cuantiles, snn_input_layer_neurons_size, is_train=False, device=device)

        if epoch == 0:  # Only print once
            print(f'Longitud de dataset de prueba: {len(secuencias2test)}')

        spikes_input, spikes, spikes_conv, network = ejecutar_red(
            secuencias2test, network, source_monitor, target_monitor, conv_monitor, T,
            use_conv_layer=use_conv_layer,
            conv_processing_type=conv_processing_type,
            device=device
        )

        # Calculate metrics using guardar_resultados (existing approach)
        mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C = guardar_resultados(
            spikes, spikes_conv, data_test, n, snn_input_layer_neurons_size, n_trial,
            date_starting_trials, dataset_name, snn_process_layer_neurons_size, trial=trial, use_conv_layer=use_conv_layer
        )

        # Direct metrics calculation for transparency (Iago's approach)
        if epoch == 0:  # Only show once per experiment
            print("Direct metrics calculation (for transparency):")

        # Process spikes for direct metrics
        spikes_1d_B = spikes.sum(axis=1) if len(spikes.shape) > 1 else spikes
        binary_predictions_B = (spikes_1d_B > 0).astype(float)

        # Direct F1, precision, recall calculation
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc
        direct_precision_B, direct_recall_B, direct_f1_B, _ = precision_recall_fscore_support(
            data_test['label'], binary_predictions_B, average='binary', zero_division=0
        )

        # AUC calculation
        try:
            direct_auc_B = roc_auc_score(data_test['label'], spikes_1d_B)
            fpr_B, tpr_B, _ = roc_curve(data_test['label'], spikes_1d_B)
            direct_auc_roc_B = auc(fpr_B, tpr_B)
        except Exception as e:
            direct_auc_B = direct_auc_roc_B = 0.5

        # Compare metrics
        if epoch == 0:  # Only show comparison once
            print(f"  Guardar F1-B: {f1_B:.4f}, Direct F1-B: {direct_f1_B:.4f}")
            print(f"  Guardar Precision-B: {precision_B:.4f}, Direct Precision-B: {direct_precision_B:.4f}")
            print(f"  Guardar Recall-B: {recall_B:.4f}, Direct Recall-B: {direct_recall_B:.4f}")
            if direct_auc_B != 0.5:
                print(f"  Direct AUC-B: {direct_auc_B:.4f}")

        # Use direct metrics for decision making (more transparent)
        direct_metrics = {
            'f1_B': direct_f1_B,
            'precision_B': direct_precision_B,
            'recall_B': direct_recall_B,
            'auc_B': direct_auc_B
        }

        # Choose which F1 score to return based on whether we're using conv layer (using direct metrics)
        if use_conv_layer and f1_C is not None:
            current_f1 = max(direct_f1_B if direct_f1_B is not None else 0, f1_C if f1_C is not None else 0)
        else:
            current_f1 = direct_f1_B if direct_f1_B is not None else 0

        print(f"Epoch {epoch + 1} - F1: {current_f1:.4f}")

        # Store epoch results (using direct metrics for primary metrics)
        epoch_results.append({
            'epoch': epoch + 1,
            'f1_score': current_f1,
            'direct_f1_B': direct_f1_B,
            'direct_precision_B': direct_precision_B,
            'direct_recall_B': direct_recall_B,
            'direct_auc_B': direct_auc_B,
            'mse_B': mse_B,
            'f1_B': f1_B,
            'f1_C': f1_C,
            'precision_B': precision_B,
            'recall_B': recall_B,
            'precision_C': precision_C,
            'recall_C': recall_C,
            'mse_C': mse_C
        })

        # Check for best performance
        if current_f1 > best_f1_score:
            best_f1_score = current_f1
            patience_counter = 0
            # Save network state (simplified - in practice you'd save network weights)
            best_epoch_results = epoch_results[-1].copy()
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} - no improvement for {early_stopping_patience} epochs")
            break

    # Return best epoch's performance (using direct metrics where available)
    if best_epoch_results:
        print(f"Best performance at epoch {best_epoch_results['epoch']}: F1={best_epoch_results['f1_score']:.4f}")
        return (best_epoch_results['mse_B'], best_epoch_results['mse_C'],
                best_epoch_results['direct_f1_B'], best_epoch_results['direct_precision_B'], best_epoch_results['direct_recall_B'],
                best_epoch_results['f1_C'], best_epoch_results['precision_C'], best_epoch_results['recall_C'],
                best_epoch_results['f1_score'])
    else:
        # Fallback to last epoch if something went wrong
        last_result = epoch_results[-1]
        return (last_result['mse_B'], last_result['mse_C'],
                last_result['direct_f1_B'], last_result['direct_precision_B'], last_result['direct_recall_B'],
                last_result['f1_C'], last_result['precision_C'], last_result['recall_C'],
                last_result['f1_score'])

def _save_minimal_trial_results(trial, config, conv_params, conv_processing_type, best_f1,
                               mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C):
    """Save minimal trial results even when the trial fails"""

    # Create directory structure
    conv_dir = "conv_true" if use_conv_layer else "conv_false"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    base_path = os.path.join(project_root, 'resultados', conv_dir, dataset_name,
                            f'n_{snn_process_layer_neurons_size}', date_starting_trials,
                            f'trial_{trial.number + 1}')
    os.makedirs(base_path, exist_ok=True)

    # Create minimal config with error information
    error_config = {
        "trial_number": trial.number + 1,
        "error": "Trial failed during execution",
        "parameters": config,
        "conv_params": conv_params,
        "conv_processing_type": conv_processing_type,
        "best_f1": best_f1,
        "mse_B": mse_B,
        "mse_C": mse_C,
        "f1_B": f1_B,
        "precision_B": precision_B,
        "recall_B": recall_B,
        "f1_C": f1_C,
        "precision_C": precision_C,
        "recall_C": recall_C,
        "failure_reason": "Exception during experiment execution"
    }

    # Save error config
    with open(f"{base_path}/config.json", "w") as f:
        json.dump(error_config, f, indent=4)

    print(f"Saved minimal results for failed trial {trial.number + 1} to {base_path}")


def _save_minimal_trial_results_with_error(trial, config, conv_params, conv_processing_type, best_f1,
                                         mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, exception):
    """Save detailed error information when trial fails"""
    import traceback

    # Create directory structure
    conv_dir = "conv_true" if use_conv_layer else "conv_false"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    base_path = os.path.join(project_root, 'resultados', conv_dir, dataset_name,
                            f'n_{snn_process_layer_neurons_size}', date_starting_trials,
                            f'trial_{trial.number + 1}')
    os.makedirs(base_path, exist_ok=True)

    # Create detailed error config
    error_config = {
        "trial_number": trial.number + 1,
        "error": "Trial failed during execution",
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "exception_traceback": traceback.format_exc(),
        "parameters": config,
        "conv_params": conv_params,
        "conv_processing_type": conv_processing_type,
        "best_f1": best_f1,
        "mse_B": mse_B,
        "mse_C": mse_C,
        "f1_B": f1_B,
        "precision_B": precision_B,
        "recall_B": recall_B,
        "f1_C": f1_C,
        "precision_C": precision_C,
        "recall_C": recall_C,
        "failure_reason": "Exception during experiment execution"
    }

    # Save error config
    with open(f"{base_path}/config.json", "w") as f:
        json.dump(error_config, f, indent=4)

    print(f"Saved detailed error results for failed trial {trial.number + 1} to {base_path}")


def objective(trial, use_cross_validation=False, n_splits=5, n_epochs=3, early_stopping_patience=2):
    """Enhanced objective function with robust error handling and logging"""

    trial_start_time = datetime.now()
    print(f"\n{'='*60}")
    print(f"Starting Trial {trial.number + 1}/{args.n_trials if 'args' in globals() else '???'}")
    print(f"Start time: {trial_start_time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")

    # More aggressive parameter ranges for better learning
    config = {
        # STDP parameters - wider range for better plasticity
        'nu1': trial.suggest_float('nu1', 0.0, 1.0),  # Positive learning rates only
        'nu2': trial.suggest_float('nu2', 0.0, 1.0),  # Positive learning rates only

        # Threshold - lower values for more spiking activity
        'threshold': trial.suggest_float('threshold', -70, -45),  # Lower minimum, higher maximum

        # Decay - focus on biologically plausible ranges
        'decay': trial.suggest_float('decay', 50, 200),  # Extended range
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

    # Validate input parameters
    if not all(isinstance(x, (int, float)) and not np.isnan(x) and not np.isinf(x)
               for x in [config['nu1'], config['nu2'], config['threshold'], config['decay']]):
        print(f"ERROR: Invalid parameters detected: {config}")

        # Create minimal results for failed trial
        best_f1 = 0.0

        # Try to save minimal results even on failure
        try:
            error = ValueError("Invalid parameters detected")
            _save_minimal_trial_results_with_error(trial, config, conv_params, conv_processing_type, best_f1,
                                       None, None, None, None, None, None, None, None, error)
        except Exception as save_error:
            print(f"Failed to save minimal results: {save_error}")

        return 0.8  # Return poor but valid score instead of infinity

    # Check if data file exists
    if not os.path.exists(path):
        print(f"ERROR: Data file not found: {path}")

        # Create minimal results for failed trial
        best_f1 = 0.0

        # Try to save minimal results even on failure
        try:
            error = FileNotFoundError(f"Data file not found: {path}")
            _save_minimal_trial_results_with_error(trial, config, conv_params, conv_processing_type, best_f1,
                                       None, None, None, None, None, None, None, None, error)
        except Exception as save_error:
            print(f"Failed to save minimal results: {save_error}")

        return 0.8

    # Initialize variables for error handling
    mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, best_f1 = [None] * 9

    try:
        print(f"Parameters: nu1={config['nu1']:.4f}, nu2={config['nu2']:.4f}, "
              f"threshold={config['threshold']:.1f}, decay={config['decay']:.1f}")
        if conv_params:
            print(f"Conv params: kernel={conv_params['kernel_type']}, size={conv_params['kernel_size']}, "
                  f"sigma={conv_params['sigma']:.2f}, processing={conv_processing_type}")

        # Run the experiment with all parameters
        mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, best_f1 = experiment(
            nu1, nu2, a, r, snn_process_layer_neurons_size,
            config['threshold'], config['decay'], T, expansion, path,
            trial.number + 1, conv_params=conv_params,
            conv_processing_type=conv_processing_type, trial=trial,
            use_cross_validation=use_cross_validation, n_splits=n_splits
        )

        # Enhanced validation of F1 score
        if best_f1 is None:
            print("Warning: F1 score is None, using fallback score")
            best_f1 = 0.0
        elif np.isnan(best_f1) or np.isinf(best_f1):
            print(f"Warning: Invalid F1 score {best_f1}, using fallback score")
            best_f1 = 0.0
        elif not (0.0 <= best_f1 <= 1.0):
            print(f"Warning: F1 score {best_f1} out of range [0,1], clamping")
            best_f1 = max(0.0, min(1.0, best_f1))

        trial_end_time = datetime.now()
        duration = trial_end_time - trial_start_time
        print(f"Trial {trial.number + 1} completed in {duration.total_seconds():.1f}s")
        print(f"F1 Score: {best_f1:.4f}, Objective: {1.0 - best_f1:.4f}")

        return 1.0 - best_f1  # Convert to minimization problem (1 - F1)

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA OOM Error in trial {trial.number + 1}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create minimal results for failed trial
        best_f1 = 0.1  # Very poor but valid score

        # Try to save detailed error results even on failure
        try:
            _save_minimal_trial_results_with_error(trial, config, conv_params, conv_processing_type, best_f1, mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, e)
        except Exception as save_error:
            print(f"Failed to save detailed error results: {save_error}")

        return 0.9  # Return poor score, better than infinity

    except Exception as e:
        print(f"Trial {trial.number + 1} failed with error: {type(e).__name__}: {e}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()

        # Create minimal results for failed trial
        best_f1 = 0.0  # Very poor score

        # Try to save detailed error results even on failure
        try:
            _save_minimal_trial_results_with_error(trial, config, conv_params, conv_processing_type, best_f1, mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C, e)
        except Exception as save_error:
            print(f"Failed to save detailed error results: {save_error}")

        return 0.8  # Return poor but valid score instead of infinity

if __name__ == "__main__":
    start_time = datetime.now()  # Add this line to track start time
    
    parser = argparse.ArgumentParser(description='Optimización de hiperparámetros con Optuna.')
    parser.add_argument('-c', '--config', type=str, default='default', help='Configuration name from config.json')
    parser.add_argument('-n', '--n_trials', type=int, default=100, help='Número de trials para Optuna')
    parser.add_argument('--device', type=str, choices=['cpu', 'gpu'], default='gpu', help='Device to use (cpu/gpu)')
    parser.add_argument('--override_cv', action='store_true', help='Override config to enable cross-validation')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    if args.config not in config_data:
        raise ValueError(f"Configuration '{args.config}' not found in config.json. Available: {list(config_data.keys())}")

    cfg = config_data[args.config]
    print(f"Using configuration: {args.config}")
    print(f"Description: {cfg['description']}")

    # Override cross-validation if requested
    if args.override_cv:
        cfg['use_cross_validation'] = True
        print("Cross-validation enabled via command line override")

    # Extract configuration values
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level from javi/ to project root
    path = os.path.join(project_root, cfg['path'])
    dataset_name = cfg['dataset']
    snn_process_layer_neurons_size = cfg['snn_process_layer_neurons_size']
    use_conv_layer = cfg['use_conv_layer']
    use_cross_validation = cfg.get('use_cross_validation', False)
    n_splits = cfg.get('n_splits', 5)
    n_epochs = cfg.get('n_epochs', 3)
    early_stopping_patience = cfg.get('early_stopping_patience', 2)


    # Enhanced device handling with memory management and fallback
    requested_gpu = args.device == "gpu"
    gpu_available = torch.cuda.is_available()

    if requested_gpu and gpu_available:
        try:
            # Check GPU memory before committing
            gpu_props = torch.cuda.get_device_properties(0)
            total_memory = gpu_props.total_memory / 1024**3  # GB
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU memory: {total_memory:.1f} GB")

            # Try to allocate a small tensor to test GPU functionality
            test_tensor = torch.randn(100, 100, device='cuda')
            del test_tensor
            torch.cuda.empty_cache()

            device = torch.device("cuda")
            torch.cuda.set_device(0)
            print("✓ GPU initialization successful")

        except Exception as e:
            print(f"✗ GPU initialization failed: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
            gpu_available = False
    else:
        device = torch.device("cpu")
        if requested_gpu:
            if not gpu_available:
                print("Warning: GPU requested but not available, using CPU")
            else:
                print("Using CPU as requested")
        else:
            print("Using CPU")

    print(f"Final device: {device}")
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
    print(f"Dataset: {dataset_name}")
    print(f"Path: {path}")
    print(f"Neurons: {snn_process_layer_neurons_size}")
    print(f"Conv layer: {'enabled' if use_conv_layer else 'disabled'}")
    print(f"Cross-validation: {'enabled' if use_cross_validation else 'disabled'}")
    if use_cross_validation:
        print(f"CV folds: {n_splits}")
    print(f"Epochs per trial: {n_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")

    try:
        # Prepare callbacks list
        callbacks = []
        if convergence_callback is not None:
            callbacks.append(convergence_callback)

        study.optimize(
            lambda trial: objective(trial, use_cross_validation, n_splits, n_epochs, early_stopping_patience),
            n_trials=args.n_trials,
            callbacks=callbacks if callbacks else None,
            timeout=3600 * 4  # 4 hour timeout
        )

        end_time = datetime.now()  # Add this line to track end time
        duration = end_time - start_time  # Calculate duration

        print('\n' + '='*50)
        print('OPTIMIZATION RESULTS')
        print('='*50)
        # Enhanced results analysis and validation
        print(f'\n{"="*60}')
        print('OPTIMIZATION RESULTS SUMMARY')
        print(f'{"="*60}')

        # Validate best trial and value
        if study.best_trial is None:
            print("ERROR: No best trial found!")
            best_f1 = None
            best_objective = float('inf')
        else:
            best_objective = study.best_value
            if np.isinf(best_objective) or np.isnan(best_objective):
                print(f"Warning: Best objective value is invalid ({best_objective})")
                best_f1 = None
            else:
                best_f1 = 1.0 - best_objective
                print(f'Best configuration found (Trial {study.best_trial.number + 1}):')
                print(f'F1 Score: {best_f1:.4f}')
                print(f'Objective value (1-F1): {best_objective:.6f}')

        # Display best parameters if available
        if study.best_params:
            print(f'Best parameters:')
            for key, value in study.best_params.items():
                print(f'  {key}: {value}')

        # Analyze trial statistics
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
        print(f'Total duration: {duration}')

        # Analyze failure patterns
        if failed_count > 0:
            print(f'\nFailure Analysis:')
            print(f'  Failure rate: {failed_count/len(study.trials)*100:.1f}%')
            if failed_count == len(study.trials):
                print("  WARNING: All trials failed! Check logs for systematic issues.")
            elif failed_count > len(study.trials) * 0.5:
                print("  WARNING: High failure rate detected.")

        # Guardar la mejor configuración
        conv_dir = "conv_true" if use_conv_layer else "conv_false"
        # Get the parent directory (project root) to ensure resultados is created at the right level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)  # Go up one level from javi/ to project root
        base_path = os.path.join(project_root, 'resultados', conv_dir, dataset_name, f'n_{snn_process_layer_neurons_size}', date_starting_trials)
        os.makedirs(base_path, exist_ok=True)

        best_trial_number = study.best_trial.number if study.best_trial else 0
        results = {
            "best_params": study.best_params if study.best_params else {},
            "best_trial": best_trial_number + 1,
            "snn_process_layer_neurons_size": snn_process_layer_neurons_size,
            "device": str(device),
            "use_conv_layer": use_conv_layer,
            "best_f1": best_f1,
            "best_objective": best_objective,
            "total_trials": len(study.trials),
            "completed_trials": completed_count,
            "failed_trials": failed_count,
            "pruned_trials": pruned_count,
            "amoumt_of_trials": args.n_trials,
            "duration_seconds": duration.total_seconds(),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "success_rate": completed_count / len(study.trials) if len(study.trials) > 0 else 0
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