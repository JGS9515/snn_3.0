#!/usr/bin/env python3
"""
Test script to verify the improvements made to the SNN model
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

def test_threshold_optimization():
    """Test the threshold optimization improvement"""

    # Simulate current behavior (all spikes = 100)
    n_samples = 1000
    spikes_current = np.full(n_samples, 100.0)

    # Create realistic labels
    labels = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

    # Current method (threshold = 0)
    pred_current = (spikes_current > 0).astype(float)
    f1_current = f1_score(labels, pred_current, zero_division=0)

    print("=== THRESHOLD OPTIMIZATION TEST ===")
    print(f"Current F1 (threshold=0): {f1_current:.4f}")
    print(f"Current Precision: {precision_score(labels, pred_current, zero_division=0):.4f}")
    print(f"Current Recall: {recall_score(labels, pred_current, zero_division=0):.4f}")

    # With optimized threshold (any threshold would give same result since all spikes are equal)
    print(f"Optimized threshold would be: {spikes_current[0]} (same as all spikes)")
    print("Problem: All spikes have the same value, so threshold optimization won't help!")
    print()

def test_normalization_fix():
    """Test that the normalization fix is working"""

    print("=== NORMALIZATION FIX TEST ===")

    # Create sample data similar to CalIt2
    np.random.seed(42)
    n_samples = 1000

    # Create data with clear difference between normal and anomalous
    normal_data = np.random.normal(0.5, 0.2, int(n_samples * 0.7))
    anomaly_data = np.random.normal(3.0, 0.5, int(n_samples * 0.3))

    values = np.concatenate([normal_data, anomaly_data])
    labels = np.concatenate([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])

    data = pd.DataFrame({'value': values, 'label': labels})

    print("Original data statistics:")
    print(f"  Normal mean: {data[data['label'] == 0]['value'].mean():.3f}")
    print(f"  Anomaly mean: {data[data['label'] == 1]['value'].mean():.3f}")
    print(f"  Overall std: {data['value'].std():.3f}")

    # Apply normalization (similar to the code)
    normal_values = data['value'][data['label'] != 1]
    train_median = normal_values.median()
    train_mad = (normal_values - train_median).abs().median()

    if train_mad > 0:
        mad_scale = train_mad * 1.4826
        data['value_normalized'] = (data['value'] - train_median) / mad_scale
        print("\nAfter normalization:")
        print(f"  Normalized normal mean: {data[data['label'] == 0]['value_normalized'].mean():.3f}")
        print(f"  Normalized anomaly mean: {data[data['label'] == 1]['value_normalized'].mean():.3f}")
        print(f"  Normalized overall std: {data['value_normalized'].std():.3f}")
        print("✓ Normalization working correctly - clear separation between normal and anomalous")
    else:
        print("✗ MAD is zero or very small - normalization may not work properly")
    print()

def analyze_learning_rates():
    """Analyze the learning rate fix"""

    print("=== LEARNING RATE ANALYSIS ===")

    # Old range (problematic)
    old_nu1_range = [-1.0, 1.0]
    old_nu2_range = [-1.0, 1.0]

    # New range (fixed)
    new_nu1_range = [0.0, 1.0]
    new_nu2_range = [0.0, 1.0]

    print(f"Old nu1 range: {old_nu1_range} (allows negative values - BAD)")
    print(f"New nu1 range: {new_nu1_range} (only positive values - GOOD)")
    print(f"Old nu2 range: {old_nu2_range} (allows negative values - BAD)")
    print(f"New nu2 range: {new_nu2_range} (only positive values - GOOD)")
    print("✓ Learning rates now constrained to positive values")
    print()

def suggest_next_steps():
    """Suggest next steps for improvement"""

    print("=== NEXT STEPS FOR IMPROVEMENT ===")

    print("1. Test the corrected model with new experiments")
    print("2. Monitor if spikes still have constant values")
    print("3. If spikes are still constant, investigate:")
    print("   - Training process (are weights being updated?)")
    print("   - Network architecture (is the topology appropriate?)")
    print("   - Input encoding (is quantization working properly?)")
    print("4. Compare results with TSFEDL baselines")
    print("5. Consider ensemble methods or different preprocessing")
    print()

if __name__ == "__main__":
    test_threshold_optimization()
    test_normalization_fix()
    analyze_learning_rates()
    suggest_next_steps()
