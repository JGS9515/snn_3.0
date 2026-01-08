#!/usr/bin/env python3
"""
Debug script to understand why spikes are constant
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def analyze_spike_patterns():
    """Analyze spike patterns from existing results"""

    # Load data from a trial
    trial_path = "resultados/conv_true/callt2/n_100/2025_08_31-23_01/trial_43"

    spikes = np.loadtxt(f'{trial_path}/spikes')
    labels = np.loadtxt(f'{trial_path}/label')
    values = np.loadtxt(f'{trial_path}/value')

    print("=== SPIKE PATTERN ANALYSIS ===")
    print(f"Data shape: {spikes.shape}")
    print(f"Unique spike values: {np.unique(spikes)}")
    print(f"Spike statistics: mean={np.mean(spikes):.2f}, std={np.std(spikes):.2f}")
    print(f"Anomaly ratio in labels: {np.mean(labels):.3f}")

    # Check if spikes vary by time windows
    if len(spikes.shape) > 1:
        print(f"Spike pattern over time (first 10 time steps):")
        for i in range(min(10, spikes.shape[1])):
            unique_vals = np.unique(spikes[:, i])
            print(f"  Time step {i}: {len(unique_vals)} unique values, range: {spikes[:, i].min()}-{spikes[:, i].max()}")

    # Check relationship between input values and spikes
    print("
Input value statistics:")
    print(f"  Range: {values.min():.2f} - {values.max():.2f}")
    print(f"  Mean: {values.mean():.2f}, Std: {values.std():.2f}")

    # Check if high input values correspond to high spikes
    high_value_indices = values > np.percentile(values, 90)
    low_value_indices = values < np.percentile(values, 10)

    if len(spikes.shape) == 1:
        print("
Spike analysis (1D):")
        print(f"  High values spikes: mean={spikes[high_value_indices].mean():.2f}")
        print(f"  Low values spikes: mean={spikes[low_value_indices].mean():.2f}")
    else:
        print("
Spike analysis (2D):")
        high_spikes = spikes[high_value_indices].mean(axis=0)
        low_spikes = spikes[low_value_indices].mean(axis=0)
        print(f"  High values spikes (mean per timestep): {high_spikes}")
        print(f"  Low values spikes (mean per timestep): {low_spikes}")

    # Load config to check parameters
    import json
    with open(f'{trial_path}/config.json', 'r') as f:
        config = json.load(f)

    print("
Model parameters:")
    print(f"  nu1: {config.get('nu1', 'N/A')}")
    print(f"  nu2: {config.get('nu2', 'N/A')}")
    print(f"  threshold: {config.get('threshold', 'N/A')}")
    print(f"  decay: {config.get('decay', 'N/A')}")

    return {
        'spike_std': np.std(spikes),
        'spike_unique_count': len(np.unique(spikes)),
        'nu1': config.get('nu1'),
        'nu2': config.get('nu2')
    }

def suggest_improvements(results):
    """Suggest improvements based on analysis"""

    print("\n=== SUGGESTED IMPROVEMENTS ===")

    if results['spike_std'] == 0:
        print("CRITICAL: All spikes have the same value!")
        print("  - Learning rates may be incorrect")
        print("  - Model may not be learning at all")
        print("  - Check if training data has proper labels")

    if results['nu1'] is not None and results['nu1'] < 0:
        print("WARNING: nu1 is negative - this prevents proper learning")

    if results['nu2'] is not None and results['nu2'] < 0:
        print("WARNING: nu2 is negative - this prevents proper learning")

    print("\nRecommendations:")
    print("1. Ensure learning rates (nu1, nu2) are positive")
    print("2. Check that training data has proper anomaly labels")
    print("3. Verify that the model architecture is suitable for the data")
    print("4. Consider adjusting threshold values")
    print("5. Try different kernel configurations for conv layer")

if __name__ == "__main__":
    results = analyze_spike_patterns()
    suggest_improvements(results)
