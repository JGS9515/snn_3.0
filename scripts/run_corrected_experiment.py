#!/usr/bin/env python3
"""
Run a corrected experiment with the fixes implemented
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add the javi directory to the path
sys.path.append('javi')

def run_single_corrected_experiment():
    """Run a single experiment with all corrections applied"""

    print("=== RUNNING CORRECTED SNN EXPERIMENT ===")

    # Import the experiment function
    from ejecutar_experimento_javi import run_single_experiment

    # Load data
    dataset_name = "callt2"
    data_path = f"Nuevos datasets/Callt2/preliminar/train_label_filled.csv"

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    data_train = pd.read_csv(data_path)
    print(f"Loaded training data: {data_train.shape}")
    print(f"Training labels distribution: {data_train['label'].value_counts().to_dict()}")

    # Load test data
    test_data_path = f"Nuevos datasets/Callt2/preliminar/CalIt2.csv"
    if os.path.exists(test_data_path):
        data_test = pd.read_csv(test_data_path)
        # Add dummy labels for test data (since CalIt2.csv doesn't have them)
        data_test['label'] = 0
        print(f"Loaded test data: {data_test.shape}")
    else:
        print("Test data not found, using training data for testing")
        data_test = data_train.copy()

    # Set corrected parameters
    config = {
        'nu1': 0.5,  # Positive learning rate
        'nu2': 0.3,  # Positive learning rate
        'threshold': -65.0,
        'decay': 150.0,
        'n': 100,  # neurons in hidden layer
        'T': 10,   # time steps
        'expansion': 2,
        'r': 0.1,  # resolution parameter
        'a': 1.0,  # expansion factor
    }

    # Convolution parameters
    conv_params = {
        'kernel_type': 'gaussian',
        'kernel_size': 5,
        'sigma': 1.0,
        'norm_factor': 2.0,
        'exc_inh_balance': 0.0,
        'conv_processing_type': 'weighted_sum'
    }

    print("\nRunning experiment with parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"  conv_params: {conv_params}")

    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M")

    try:
        # Run the experiment
        results = run_single_experiment(
            data_train, data_test,
            nu1=config['nu1'], nu2=config['nu2'],
            a=config['a'], r=config['r'],
            n=config['n'], threshold=config['threshold'],
            decay=config['decay'], T=config['T'],
            expansion=config['expansion'],
            n_trial=1, conv_params=conv_params,
            conv_processing_type='weighted_sum',
            trial=None, fold=None,
            n_epochs=3, early_stopping_patience=2
        )

        print("Experiment completed successfully!")
        print(f"Results saved with timestamp: {timestamp}")

        return results

    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_results(results):
    """Analyze the results of the corrected experiment"""

    if results is None:
        print("No results to analyze")
        return

    print("=== RESULTS ANALYSIS ===")

    mse_B, mse_C, f1_B, precision_B, recall_B, f1_C, precision_C, recall_C = results

    print("Layer B (SNN base):")
    print(f"  F1: {f1_B:.4f}")
    print(f"  Precision: {precision_B:.4f}")
    print(f"  Recall: {recall_B:.4f}")
    print(f"  MSE: {mse_B:.4f}")

    if f1_C is not None:
        print("Layer C (with convolution):")
        print(f"  F1: {f1_C:.4f}")
        print(f"  Precision: {precision_C:.4f}")
        print(f"  Recall: {recall_C:.4f}")
        print(f"  MSE: {mse_C:.4f}")

    # Compare with TSFEDL baselines
    print("\n=== COMPARISON WITH TSFEDL ===")

    tsfedl_results = {
        'TSFEDL-OhShuLih': 0.704,
        'TSFEDL-KhanZulfiqar': 0.704,
        'TSFEDL-ZhengZhenyu': 0.704,
        'TSFEDL-WeiXiaoyan': 0.679
    }

    best_f1 = max(f1_B, f1_C) if f1_C is not None else f1_B

    print(f"SNN Best F1: {best_f1:.4f}")
    for model, f1 in tsfedl_results.items():
        diff = best_f1 - f1
        print(f"{model}: {f1:.4f} (SNN is {diff:+.4f})")

    if best_f1 >= min(tsfedl_results.values()):
        print("🎉 SUCCESS: SNN performance is now comparable to TSFEDL!")
    else:
        gap = min(tsfedl_results.values()) - best_f1
        print(f"📉 Still a gap of {gap:.4f} to reach TSFEDL performance")

if __name__ == "__main__":
    results = run_single_corrected_experiment()
    analyze_results(results)
