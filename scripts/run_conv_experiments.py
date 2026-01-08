#!/usr/bin/env python3
"""
Script to run comprehensive SNN experiments comparing conv vs non-conv layers
for different neuron sizes: 100, 200, 400 with 100 trials each.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_experiment(neuron_size, use_conv_layer, n_trials=100, device='cpu'):
    """Run a single experiment configuration by directly modifying and running the script."""
    
    # Backup the original file
    script_path = "javi/ejecutar_experimento_javi.py"
    backup_path = "javi/ejecutar_experimento_javi_backup.py"
    
    # Read the original file
    with open(script_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Save backup
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)
    
    # Modify the configuration in the script
    modified_content = original_content
    
    # Update the neuron size
    modified_content = modified_content.replace(
        'snn_process_layer_neurons_size=100',
        f'snn_process_layer_neurons_size={neuron_size}'
    )
    
    # Update the conv layer setting
    modified_content = modified_content.replace(
        'use_conv_layer=True',
        f'use_conv_layer={use_conv_layer}'
    )
    
    # Write the modified content
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"\n{'='*80}")
    print(f"🧪 Running experiment:")
    print(f"  Neuron size: {neuron_size}")
    print(f"  Conv layer: {'conv_true' if use_conv_layer else 'conv_false'}")
    print(f"  Trials: {n_trials}")
    print(f"  Device: {device}")
    print(f"{'='*80}")
    
    try:
        # Run the experiment
        cmd = [sys.executable, script_path, '--n_trials', str(n_trials), '--device', device]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print(f"✅ Successfully completed: n_{neuron_size}/{'conv_true' if use_conv_layer else 'conv_false'}")
            print("Output (last 500 chars):", result.stdout[-500:])
        else:
            print(f"❌ Failed: n_{neuron_size}/{'conv_true' if use_conv_layer else 'conv_false'}")
            print("STDERR:", result.stderr)
            print("STDOUT:", result.stdout[-1000:])
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Exception during experiment: {e}")
        return False
        
    finally:
        # Restore the original file
        with open(backup_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        # Remove backup
        if os.path.exists(backup_path):
            os.remove(backup_path)

def main():
    """Run all experiment configurations."""
    
    # Configuration
    neuron_sizes = [100, 200, 400]
    conv_configs = [False, True]  # conv_false, conv_true
    n_trials = 100
    device = 'cpu'  # Change to 'gpu' if you have CUDA available
    
    print("🚀 Starting comprehensive SNN conv layer comparison")
    print(f"Neuron sizes: {neuron_sizes}")
    print(f"Conv configurations: {['conv_false', 'conv_true']}")
    print(f"Trials per config: {n_trials}")
    print(f"Device: {device}")
    
    # Track results
    results = []
    total_experiments = len(neuron_sizes) * len(conv_configs)
    current_experiment = 0
    
    start_time = datetime.now()
    
    # Run all combinations
    for neuron_size in neuron_sizes:
        for use_conv_layer in conv_configs:
            current_experiment += 1
            
            print(f"\n📊 Experiment {current_experiment}/{total_experiments}")
            
            success = run_experiment(
                neuron_size=neuron_size,
                use_conv_layer=use_conv_layer,
                n_trials=n_trials,
                device=device
            )
            
            results.append({
                'neuron_size': neuron_size,
                'use_conv_layer': use_conv_layer,
                'success': success,
                'timestamp': datetime.now().isoformat()
            })
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total duration: {duration}")
    print(f"Experiments completed: {sum(1 for r in results if r['success'])}/{len(results)}")
    
    # Print detailed results
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        conv_str = "conv_true" if result['use_conv_layer'] else "conv_false"
        print(f"  • n_{result['neuron_size']}/{conv_str}: {status}")
    
    print(f"\n🎯 Results will be organized in:")
    for neuron_size in neuron_sizes:
        for use_conv_layer in conv_configs:
            conv_str = "conv_true" if use_conv_layer else "conv_false"
            print(f"  resultados/{conv_str}/iops/n_{neuron_size}/")

if __name__ == "__main__":
    main()
