#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_config_file(config_path):
    """Parse the parameters.txt file to extract configuration."""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line and not line.startswith('#'):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert to numeric if possible
                    try:
                        if '.' in value:
                            config[key] = float(value)
                        else:
                            config[key] = int(value)
                    except ValueError:
                        config[key] = value
    except Exception as e:
        print(f"Error reading config {config_path}: {e}")
    
    return config

def analyze_fitness_history(fitness_path):
    """Analyze a fitness history file."""
    try:
        with open(fitness_path, 'r') as f:
            data = json.load(f)
        
        if not data:
            return None
        
        # Extract key metrics
        generations = [gen['generation'] for gen in data]
        
        # Get fitness statistics for each generation
        fitness_stats = []
        for gen_data in data:
            fitness_values = gen_data['fitness']
            if fitness_values:
                fitness_stats.append({
                    'generation': gen_data['generation'],
                    'max_fitness': max(fitness_values),
                    'min_fitness': min(fitness_values),
                    'mean_fitness': np.mean(fitness_values),
                    'std_fitness': np.std(fitness_values),
                    'population_size': len(fitness_values)
                })
        
        if not fitness_stats:
            return None
        
        # Overall training summary
        final_stats = fitness_stats[-1]
        initial_stats = fitness_stats[0]
        
        summary = {
            'total_generations': len(generations),
            'max_generation': max(generations),
            'final_max_fitness': final_stats['max_fitness'],
            'final_mean_fitness': final_stats['mean_fitness'],
            'initial_max_fitness': initial_stats['max_fitness'],
            'initial_mean_fitness': initial_stats['mean_fitness'],
            'fitness_improvement': final_stats['max_fitness'] - initial_stats['max_fitness'],
            'mean_improvement': final_stats['mean_fitness'] - initial_stats['mean_fitness'],
            'population_size': final_stats['population_size'],
            'converged': final_stats['max_fitness'] > 0.95,  # Adjust threshold as needed
            'training_completed': True
        }
        
        return summary, fitness_stats
        
    except Exception as e:
        print(f"  Error analyzing {fitness_path}: {e}")
        return None

def scan_experiment_directories(root_dir):
    """Scan all experiment directories and extract data."""
    
    # Use Path and handle the path properly
    root_path = Path(root_dir).expanduser().resolve()
    print(f"Scanning root directory: {root_path}")
    
    # Use os.listdir instead of glob to avoid issues with spaces
    try:
        all_items = os.listdir(root_path)
        config_dirs = [item for item in all_items if item.startswith('config_')]
        config_dirs = sorted(config_dirs)  # Sort for consistent processing
        
        print(f"Found {len(config_dirs)} config directories")
        
    except Exception as e:
        print(f"Error listing directory: {e}")
        return [], []
    
    all_summaries = []
    all_detailed_data = []
    
    # Process in batches and show progress
    total_dirs = len(config_dirs)
    batch_size = 100
    
    for i, config_name in enumerate(config_dirs):
        config_dir = root_path / config_name
        
        # Show progress every 100 directories
        if (i + 1) % batch_size == 0 or i == 0:
            print(f"Processing {config_name}... ({i+1}/{total_dirs}) - {((i+1)/total_dirs*100):.1f}%")
        
        # Parse configuration
        config_file = config_dir / "config" / "parameters.txt"
        config_params = parse_config_file(config_file) if config_file.exists() else {}
        
        # Find fitness history file
        results_dir = config_dir / "results"
        if not results_dir.exists():
            summary = {
                'config_name': config_name,
                'training_completed': False,
                'error_reason': 'No results directory found'
            }
            summary.update(config_params)
            all_summaries.append(summary)
            continue
            
        # Find fitness files using os.listdir to avoid glob issues
        try:
            results_files = os.listdir(results_dir)
            fitness_files = [f for f in results_files if f.startswith('fitness_history_') and f.endswith('.json')]
        except:
            fitness_files = []
        
        if not fitness_files:
            summary = {
                'config_name': config_name,
                'training_completed': False,
                'error_reason': 'No fitness history file found'
            }
            summary.update(config_params)
            all_summaries.append(summary)
            continue
        
        fitness_file = results_dir / fitness_files[0]  # Take the first one if multiple
        
        # Analyze fitness history
        result = analyze_fitness_history(fitness_file)
        
        if result is None:
            summary = {
                'config_name': config_name,
                'training_completed': False,
                'error_reason': 'Error parsing fitness file'
            }
            summary.update(config_params)
            all_summaries.append(summary)
            continue
        
        summary, detailed_stats = result
        
        # Add config info to summary
        summary['config_name'] = config_name
        summary.update(config_params)
        
        all_summaries.append(summary)
        
        # Add detailed generation data (but maybe skip this for very large datasets)
        # Comment out the next few lines if you don't need generation-by-generation data
        for gen_stat in detailed_stats:
            gen_stat['config_name'] = config_name
            gen_stat.update(config_params)
            all_detailed_data.append(gen_stat)
    
    return all_summaries, all_detailed_data

def main():
    parser = argparse.ArgumentParser(description='Analyze XOR training experiments')
    parser.add_argument('--root-dir', '-r', default='.', 
                       help='Root directory containing config_* folders')
    parser.add_argument('--output-prefix', '-o', default='xor_analysis',
                       help='Prefix for output files')
    parser.add_argument('--summary-only', '-s', action='store_true',
                       help='Only generate summary data (skip detailed generation data)')
    
    args = parser.parse_args()
    
    print("Scanning experiment directories...")
    if args.summary_only:
        print("Summary-only mode: skipping detailed generation data")
    
    summaries, detailed_data = scan_experiment_directories(args.root_dir)
    
    # Convert to DataFrames
    summary_df = pd.DataFrame(summaries)
    
    # Save summary file
    summary_file = f"{args.output_prefix}_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Save detailed file only if we have data and not in summary-only mode
    if detailed_data and not args.summary_only:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = f"{args.output_prefix}_detailed.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"Detailed data saved to: {detailed_file}")
    elif args.summary_only:
        print("Detailed data skipped (summary-only mode)")
    
    # Print quick summary
    total_configs = len(summaries)
    completed = sum(1 for s in summaries if s.get('training_completed', False))
    converged = sum(1 for s in summaries if s.get('converged', False))
    
    print(f"\nQuick Summary:")
    print(f"Total configurations: {total_configs}")
    print(f"Completed training: {completed} ({completed/total_configs*100:.1f}%)")
    print(f"Converged (>95% fitness): {converged} ({converged/total_configs*100:.1f}%)")
    
    if completed > 0:
        completed_summaries = [s for s in summaries if s.get('training_completed', False)]
        avg_final_fitness = np.mean([s['final_max_fitness'] for s in completed_summaries])
        print(f"Average final max fitness: {avg_final_fitness:.3f}")

if __name__ == "__main__":
    main()
    
    