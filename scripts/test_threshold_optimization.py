#!/usr/bin/env python3
"""
Simple test script for threshold optimization
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def find_optimal_threshold_f1(spikes_1d, ground_truth_labels):
    """Find optimal threshold by maximizing F1 score"""
    # Generate range of potential thresholds
    min_val, max_val = np.min(spikes_1d), np.max(spikes_1d)
    if min_val == max_val:
        return min_val  # All values are the same

    thresholds = np.linspace(min_val, max_val, 100)
    best_f1 = 0
    best_threshold = min_val

    for threshold in thresholds:
        binary_pred = (spikes_1d > threshold).astype(float)
        f1 = f1_score(ground_truth_labels, binary_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def test_threshold_optimization():
    """Test the threshold optimization with sample data"""

    # Create sample data similar to what we saw in the results
    np.random.seed(42)

    # Simulate spikes data (similar to what we saw in results)
    n_samples = 5250
    spikes_1d = np.random.exponential(0.5, n_samples)  # Exponential distribution

    # Create ground truth with anomaly ratio similar to CalIt2
    anomaly_ratio = 0.263
    n_anomalies = int(n_samples * anomaly_ratio)

    ground_truth = np.zeros(n_samples)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    ground_truth[anomaly_indices] = 1

    # Make anomalies have higher spikes (more realistic)
    spikes_1d[anomaly_indices] *= 2.5

    print("Testing threshold optimization...")
    print(f"Dataset size: {n_samples}")
    print(f"Anomaly ratio: {anomaly_ratio:.3f}")
    print(f"Anomalies: {n_anomalies}")

    # Test current method (threshold = 0)
    binary_pred_current = (spikes_1d > 0).astype(float)
    f1_current = f1_score(ground_truth, binary_pred_current, zero_division=0)
    precision_current = precision_score(ground_truth, binary_pred_current, zero_division=0)
    recall_current = recall_score(ground_truth, binary_pred_current, zero_division=0)

    print("
Current method (threshold = 0):")
    print(f"  F1: {f1_current:.4f}")
    print(f"  Precision: {precision_current:.4f}")
    print(f"  Recall: {recall_current:.4f}")

    # Test optimized threshold
    optimal_threshold, f1_optimized = find_optimal_threshold_f1(spikes_1d, ground_truth)
    binary_pred_optimized = (spikes_1d > optimal_threshold).astype(float)
    precision_optimized = precision_score(ground_truth, binary_pred_optimized, zero_division=0)
    recall_optimized = recall_score(ground_truth, binary_pred_optimized, zero_division=0)

    print("
Optimized threshold:")
    print(f"  Threshold: {optimal_threshold:.4f}")
    print(f"  F1: {f1_optimized:.4f}")
    print(f"  Precision: {precision_optimized:.4f}")
    print(f"  Recall: {recall_optimized:.4f}")
    print(f"  Improvement: {f1_optimized - f1_current:.4f}")

    return {
        'current_f1': f1_current,
        'optimized_f1': f1_optimized,
        'improvement': f1_optimized - f1_current,
        'optimal_threshold': optimal_threshold
    }

if __name__ == "__main__":
    results = test_threshold_optimization()

    # Test with real data from results if available
    try:
        import os
        import json

        # Try to load real data from results
        trial_path = "resultados/conv_true/callt2/n_100/2025_08_31-23_01/trial_43"

        if os.path.exists(trial_path):
            print("\n" + "="*50)
            print("Testing with REAL data from results...")

            spikes = np.loadtxt(f'{trial_path}/spikes')
            labels = np.loadtxt(f'{trial_path}/label')

            spikes_1d = spikes.sum(axis=1) if len(spikes.shape) > 1 else spikes

            # Current method
            binary_pred_current = (spikes_1d > 0).astype(float)
            f1_current = f1_score(labels, binary_pred_current, zero_division=0)

            # Optimized method
            optimal_threshold, f1_optimized = find_optimal_threshold_f1(spikes_1d, labels)

            print("
Real data results:")
            print(f"  Current F1: {f1_current:.4f}")
            print(f"  Optimized F1: {f1_optimized:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            print(f"  Improvement: {f1_optimized - f1_current:.4f}")

    except Exception as e:
        print(f"\nCould not test with real data: {e}")

