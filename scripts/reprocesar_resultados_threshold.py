#!/usr/bin/env python3
"""
Script to reprocess existing results with optimized threshold
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from javi.utils import find_optimal_threshold_f1

def reprocess_trial_results(trial_path):
    """Reprocess a single trial with optimized threshold"""

    # Load data
    try:
        spikes = np.loadtxt(f'{trial_path}/spikes')
        labels = np.loadtxt(f'{trial_path}/label')
        values = np.loadtxt(f'{trial_path}/value')

        # Check if conv results exist
        conv_exists = os.path.exists(f'{trial_path}/spikes_conv')
        if conv_exists:
            spikes_conv = np.loadtxt(f'{trial_path}/spikes_conv')
    except Exception as e:
        print(f"Error loading data from {trial_path}: {e}")
        return None

    # Load config
    with open(f'{trial_path}/config.json', 'r') as f:
        config = json.load(f)

    # Process layer B
    spikes_1d_B = spikes.sum(axis=1) if len(spikes.shape) > 1 else spikes
    optimal_threshold_B = find_optimal_threshold_f1(spikes_1d_B, labels)
    binary_predictions_B = (spikes_1d_B > optimal_threshold_B).astype(float)

    # Calculate new metrics for layer B
    new_f1_B = f1_score(labels, binary_predictions_B, zero_division=0)
    new_precision_B = precision_score(labels, binary_predictions_B, zero_division=0)
    new_recall_B = recall_score(labels, binary_predictions_B, zero_division=0)
    new_mse_B = mean_squared_error(labels, binary_predictions_B)

    results = {
        'original_f1_B': config.get('f1_B', config.get('best_f1', 0)),
        'new_f1_B': new_f1_B,
        'original_precision_B': config.get('precision_B', 0),
        'new_precision_B': new_precision_B,
        'original_recall_B': config.get('recall_B', 0),
        'new_recall_B': new_recall_B,
        'optimal_threshold_B': optimal_threshold_B,
        'improvement_B': new_f1_B - config.get('f1_B', config.get('best_f1', 0))
    }

    # Process layer C if exists
    if conv_exists and 'f1_C' in config:
        spikes_1d_C = spikes_conv.sum(axis=1) if len(spikes_conv.shape) > 1 else spikes_conv
        optimal_threshold_C = find_optimal_threshold_f1(spikes_1d_C, labels)
        binary_predictions_C = (spikes_1d_C > optimal_threshold_C).astype(float)

        new_f1_C = f1_score(labels, binary_predictions_C, zero_division=0)
        new_precision_C = precision_score(labels, binary_predictions_C, zero_division=0)
        new_recall_C = recall_score(labels, binary_predictions_C, zero_division=0)
        new_mse_C = mean_squared_error(labels, binary_predictions_C)

        results.update({
            'original_f1_C': config.get('f1_C', 0),
            'new_f1_C': new_f1_C,
            'original_precision_C': config.get('precision_C', 0),
            'new_precision_C': new_precision_C,
            'original_recall_C': config.get('recall_C', 0),
            'new_recall_C': new_recall_C,
            'optimal_threshold_C': optimal_threshold_C,
            'improvement_C': new_f1_C - config.get('f1_C', 0)
        })

    return results

def main():
    datasets = ['iops', 'callt2']
    n_values = [100, 200, 400]

    for dataset in datasets:
        print(f"\n=== Processing {dataset.upper()} ===")

        for n in n_values:
            print(f"\nProcessing n={n}")

            results_path = f'resultados/conv_true/{dataset}/n_{n}'
            if not os.path.exists(results_path):
                print(f"Path {results_path} does not exist")
                continue

            # Find experiment directories
            experiments = [d for d in os.listdir(results_path)
                          if os.path.isdir(os.path.join(results_path, d)) and d != '__pycache__']

            for exp in experiments:
                exp_path = os.path.join(results_path, exp)
                best_config_path = os.path.join(exp_path, 'best_config.json')

                if not os.path.exists(best_config_path):
                    continue

                print(f"Processing experiment: {exp}")

                # Load best config
                with open(best_config_path, 'r') as f:
                    best_config = json.load(f)

                trial_num = best_config.get('best_trial', 1)
                trial_path = os.path.join(exp_path, f'trial_{trial_num}')

                if not os.path.exists(trial_path):
                    print(f"Trial path {trial_path} does not exist")
                    continue

                # Reprocess results
                results = reprocess_trial_results(trial_path)

                if results:
                    print(f"  Original F1-B: {results['original_f1_B']:.4f}")
                    print(f"  New F1-B: {results['new_f1_B']:.4f}")
                    print(f"  Improvement: {results['improvement_B']:.4f}")
                    print(f"  Optimal threshold: {results['optimal_threshold_B']:.4f}")

                    if 'new_f1_C' in results:
                        print(f"  Original F1-C: {results['original_f1_C']:.4f}")
                        print(f"  New F1-C: {results['new_f1_C']:.4f}")
                        print(f"  Improvement: {results['improvement_C']:.4f}")
                        print(f"  Optimal threshold: {results['optimal_threshold_C']:.4f}")

if __name__ == "__main__":
    main()
