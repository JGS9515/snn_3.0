#!/usr/bin/env python3
"""
Generate LaTeX tables with the best results from all experiments.
This script analyzes all trial results and creates comprehensive tables.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def find_best_result_in_experiment(experiment_dir):
    """Find the best F1 score and corresponding metrics from all trials in an experiment."""
    
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.exists():
        return None
    
    trial_dirs = [d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')]
    
    best_result = {
        'f1_B': 0.0, 'precision_B': 0.0, 'recall_B': 0.0, 'mse_B': 0.0,
        'f1_C': 0.0, 'precision_C': 0.0, 'recall_C': 0.0, 'mse_C': 0.0,
        'best_f1': 0.0, 'best_layer': 'B', 'trial_num': None
    }
    
    for trial_dir in trial_dirs:
        config_path = trial_dir / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Check layer B
                f1_B = config.get('f1_B', 0.0)
                if f1_B > best_result['f1_B']:
                    best_result['f1_B'] = f1_B
                    best_result['precision_B'] = config.get('precision_B', 0.0)
                    best_result['recall_B'] = config.get('recall_B', 0.0)
                    best_result['mse_B'] = config.get('mse_B', 0.0)
                    best_result['best_f1'] = f1_B
                    best_result['best_layer'] = 'B'
                    best_result['trial_num'] = int(trial_dir.name.split('_')[1])
                
                # Check layer C (if exists)
                f1_C = config.get('f1_C', 0.0)
                if f1_C and f1_C > best_result['best_f1']:
                    best_result['f1_C'] = f1_C
                    best_result['precision_C'] = config.get('precision_C', 0.0)
                    best_result['recall_C'] = config.get('recall_C', 0.0)
                    best_result['mse_C'] = config.get('mse_C', 0.0)
                    best_result['best_f1'] = f1_C
                    best_result['best_layer'] = 'C'
                    best_result['trial_num'] = int(trial_dir.name.split('_')[1])
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error reading {config_path}: {e}")
                continue
    
    return best_result if best_result['trial_num'] is not None else None

def scan_all_experiments(results_base_dir):
    """Scan all experiments and collect best results."""
    
    results_base_dir = Path(results_base_dir)
    all_results = {}
    
    # Structure: conv_false/true -> dataset -> n_size -> experiment_date
    for conv_type in ['conv_false', 'conv_true']:
        conv_dir = results_base_dir / conv_type
        if not conv_dir.exists():
            continue
            
        for dataset in ['callt2', 'iops']:
            dataset_dir = conv_dir / dataset
            if not dataset_dir.exists():
                continue
                
            for n_size in ['n_100', 'n_200', 'n_400']:
                n_dir = dataset_dir / n_size
                if not n_dir.exists():
                    continue
                
                # Find the most recent experiment (latest date)
                experiment_dirs = [d for d in n_dir.iterdir() if d.is_dir()]
                if not experiment_dirs:
                    continue
                
                # Sort by directory name (which includes date) to get the latest
                latest_experiment = sorted(experiment_dirs, key=lambda x: x.name)[-1]
                
                print(f"Analyzing: {conv_type}/{dataset}/{n_size}/{latest_experiment.name}")
                
                best_result = find_best_result_in_experiment(latest_experiment)
                if best_result:
                    key = (conv_type, dataset, n_size.replace('n_', ''))
                    all_results[key] = best_result
                    print(f"  Best F1: {best_result['best_f1']:.3f} (Layer {best_result['best_layer']}, Trial {best_result['trial_num']})")
                else:
                    print(f"  No valid results found")
    
    return all_results

def format_metric(value, is_percentage=True):
    """Format a metric value for LaTeX table."""
    if value is None or value == 0.0:
        return "--"
    if is_percentage:
        return f"{value:.3f}"
    else:
        return f"{value:.3f}"

def generate_latex_table(results, dataset_name):
    """Generate LaTeX table for a specific dataset."""
    
    dataset_key = dataset_name.lower()
    
    # Collect data for the table
    rows = []
    
    # SNN (A--B) - conv_false results
    for n in ['100', '200', '400']:
        key = ('conv_false', dataset_key, n)
        if key in results:
            result = results[key]
            if result['best_layer'] == 'B':
                prec = format_metric(result['precision_B'])
                rec = format_metric(result['recall_B'])
                f1 = format_metric(result['f1_B'])
                mse = format_metric(result['mse_B'])
            else:
                prec = rec = f1 = mse = "--"
        else:
            prec = rec = f1 = mse = "--"
        
        rows.append(f"SNN (A--B) & {n} & {prec} & {rec} & {f1} & {mse} \\\\")
    
    # SNN (A--B--C) - conv_true results
    for n in ['100', '200', '400']:
        key = ('conv_true', dataset_key, n)
        if key in results:
            result = results[key]
            # For conv_true, prefer layer C if available, otherwise use B
            if result['best_layer'] == 'C' and result['f1_C'] > 0:
                prec = format_metric(result['precision_C'])
                rec = format_metric(result['recall_C'])
                f1 = format_metric(result['f1_C'])
                mse = format_metric(result['mse_C'])
            else:
                prec = format_metric(result['precision_B'])
                rec = format_metric(result['recall_B'])
                f1 = format_metric(result['f1_B'])
                mse = format_metric(result['mse_B'])
        else:
            prec = rec = f1 = mse = "--"
        
        rows.append(f"SNN (A--B--C) & {n} & {prec} & {rec} & {f1} & {mse} \\\\")
    
    # Generate the complete table
    table_content = f"""\\begin{{table}}[htbp]
\\centering
\\small
\\begin{{tabular}}{{lccccc}}
\\hline\\hline
\\textbf{{Modelo}} & \\textbf{{N}} & \\textbf{{Prec}} & \\textbf{{Rec}} & \\textbf{{F1}} & \\textbf{{MSE}} \\\\
\\hline
{rows[0]}
{rows[1]}
{rows[2]}
\\hline
{rows[3]}
{rows[4]}
{rows[5]}
\\hline\\hline
\\end{{tabular}}
\\caption{{Resultados de detección de anomalías en dataset {dataset_name.upper()}. N representa el número de neuronas en las capas B y C para los modelos SNN. Se muestran los mejores resultados obtenidos tras optimización con Optuna para cada configuración.}}
\\label{{tab:resultados-{dataset_key}-escalabilidad}}
\\end{{table}}"""

    return table_content

def generate_summary_table(results):
    """Generate a summary table with all results."""
    
    print("\n" + "="*80)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("="*80)
    
    datasets = ['callt2', 'iops']
    conv_types = [('conv_false', 'A--B'), ('conv_true', 'A--B--C')]
    n_sizes = ['100', '200', '400']
    
    for dataset in datasets:
        print(f"\n--- {dataset.upper()} DATASET ---")
        print(f"{'Model':<12} {'N':<4} {'Prec':<8} {'Rec':<8} {'F1':<8} {'MSE':<8} {'Layer':<6} {'Trial':<6}")
        print("-" * 60)
        
        for conv_type, model_name in conv_types:
            for n in n_sizes:
                key = (conv_type, dataset, n)
                if key in results:
                    result = results[key]
                    best_layer = result['best_layer']
                    
                    if best_layer == 'B':
                        prec = result['precision_B']
                        rec = result['recall_B']
                        f1 = result['f1_B']
                        mse = result['mse_B']
                    else:
                        prec = result['precision_C']
                        rec = result['recall_C']
                        f1 = result['f1_C']
                        mse = result['mse_C']
                    
                    prec_str = f"{prec:.3f}" if prec > 0 else "--"
                    rec_str = f"{rec:.3f}" if rec > 0 else "--"
                    f1_str = f"{f1:.3f}" if f1 > 0 else "--"
                    mse_str = f"{mse:.3f}" if mse > 0 else "--"
                    
                    print(f"SNN ({model_name:<6}) {n:<4} {prec_str:<8} {rec_str:<8} {f1_str:<8} {mse_str:<8} {best_layer:<6} {result['trial_num']:<6}")
                else:
                    print(f"SNN ({model_name:<6}) {n:<4} {'--':<8} {'--':<8} {'--':<8} {'--':<8} {'--':<6} {'--':<6}")

def main():
    """Main function to analyze all results and generate tables."""
    
    results_base_dir = r"C:\Users\Javier\Documents\GitHub\snn_3.0\resultados"
    
    print("Scanning all experiment results...")
    all_results = scan_all_experiments(results_base_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    # Generate summary
    generate_summary_table(all_results)
    
    # Generate LaTeX tables
    print("\n" + "="*80)
    print("LATEX TABLES")
    print("="*80)
    
    for dataset in ['callt2', 'iops']:
        table = generate_latex_table(all_results, dataset)
        print(f"\n--- {dataset.upper()} TABLE ---")
        print(table)
        
        # Save to file
        filename = f"tabla_resultados_{dataset}.tex"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(table)
        print(f"\nTable saved to: {filename}")
    
    # Save detailed results to CSV
    csv_data = []
    for (conv_type, dataset, n_size), result in all_results.items():
        model_name = "SNN (A--B)" if conv_type == 'conv_false' else "SNN (A--B--C)"
        layer = result['best_layer']
        
        if layer == 'B':
            prec, rec, f1, mse = result['precision_B'], result['recall_B'], result['f1_B'], result['mse_B']
        else:
            prec, rec, f1, mse = result['precision_C'], result['recall_C'], result['f1_C'], result['mse_C']
        
        csv_data.append({
            'dataset': dataset,
            'model': model_name,
            'neurons': n_size,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'mse': mse,
            'best_layer': layer,
            'trial': result['trial_num']
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv('resultados_completos.csv', index=False)
    print(f"\nDetailed results saved to: resultados_completos.csv")

if __name__ == "__main__":
    main()
