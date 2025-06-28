import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def generate_random_data(n_samples, random_seed=42):
    """
    Generate completely random 2D data with random binary labels.
    
    Args:
        n_samples: Number of data points to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with x, y features and random binary labels
    """
    np.random.seed(random_seed)
    
    # Generate random 2D points in range [-1, 1]
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # Assign completely random binary labels
    labels = np.random.randint(0, 2, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'label': labels
    })
    
    return df

def plot_random_data(df, output_path=None):
    """Plot the generated random data for visualization"""
    plt.figure(figsize=(10, 8))
    
    # Separate classes
    class_0 = df[df['label'] == 0]
    class_1 = df[df['label'] == 1]
    
    # Plot each class
    plt.scatter(class_0['x'], class_0['y'], c='blue', alpha=0.6, label='Class 0')
    plt.scatter(class_1['x'], class_1['y'], c='red', alpha=0.6, label='Class 1')
    
    plt.title(f'Random 2D Data ({len(df)} points)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def generate_datasets(output_dir='data/raw', sizes=None, random_seed=42):
    """Generate multiple datasets of different sizes"""
    if sizes is None:
        sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    for size in sizes:
        # Generate data
        df = generate_random_data(n_samples=size, random_seed=random_seed)
        
        # Create filename
        filename = f"random_data_{size}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Generated {size} random points saved to {filepath}")
        
        # Plot
        plot_path = filepath.replace('.csv', '.png').replace('/raw/', '/plots/')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plot_random_data(df, plot_path)
        
        generated_files.append(filepath)
    
    return generated_files

def generate_config_files(output_dir='config/yaml', datasets=None):
    """Generate configuration files for different network architectures"""
    os.makedirs(output_dir, exist_ok=True)
    
    if datasets is None:
        dataset_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    else:
        dataset_sizes = [int(os.path.basename(f).split('_')[2].split('.')[0]) for f in datasets]
    
    # Different network architectures to test
    hidden_layers_options = [0, 1, 2, 3, 4, 5, 10, 20]  # 0 means linear model
    neurons_per_layer_options = [2, 4, 8, 16, 32, 64, 128]
    
    # Standard GA parameters
    ga_params = {
        'population_size': 100,
        'cxpb': 0.8,
        'mutpb': 0.2,
        'ngen': 100,
        'n_processes': 4,
        'max_time_per_ind': 7200,
        'epochs': 20
    }
    
    # Standard model parameters
    model_params = {
        'activation': 'relu',
        'optimizer': 'adam',
        'lr': 0.001,
        'batch_size': 32,
        'input_dim': 2  # 2D data
    }
    
    # Generate configs
    timestamp = datetime.now().strftime("%Y%m%d")
    config_counter = 0
    
    for size in dataset_sizes:
        for hidden_layers in hidden_layers_options:
            for neurons in neurons_per_layer_options:
                # Skip very large networks for very small datasets
                if size < 50 and (hidden_layers > 3 or neurons > 32):
                    continue
                
                # Create model hidden layers configuration
                if hidden_layers == 0:
                    # Linear model has no hidden layers
                    hidden_layer_config = []
                else:
                    hidden_layer_config = [neurons] * hidden_layers
                
                # Create configuration
                config = {
                    'experiment': {
                        'id': f"random_{size}_h{hidden_layers}_n{neurons}",
                        'description': f"Random data ({size} points) with {hidden_layers} hidden layers of {neurons} neurons",
                        'noise_dimensions': 0,
                        'class_separation': 0,
                        'use_gpu': True
                    },
                    'data': {
                        'input_dim': 2,
                        'class_distribution': 0.5,
                        'dataset_size': size,
                        'use_gpu': True
                    },
                    'ga': ga_params,
                    'model': {
                        'hidden_layers': hidden_layer_config,
                        'neurons_per_layer': neurons,
                        'skip_connections': None,
                        'activation': model_params['activation'],
                        'optimizer': model_params['optimizer'],
                        'lr': model_params['lr'],
                        'batch_size': min(32, size//2) if size > 2 else 1,
                        'input_dim': 2
                    },
                    'metrics': {
                        'tracking': {
                            'evaluation': {
                                'track_accuracy': True,
                                'track_loss': True,
                                'track_success_probability': True
                            },
                            'model': {
                                'track_capacity_utilization': True,
                                'track_parameters': True
                            },
                            'training': {
                                'track_convergence': True,
                                'track_memory': True,
                                'track_time': True
                            }
                        }
                    },
                }
                
                # Generate filename
                filename = f"config_{timestamp}_{config_counter:03d}_random_{size}_h{hidden_layers}_n{neurons}.yaml"
                filepath = os.path.join(output_dir, filename)
                
                # Save configuration
                with open(filepath, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f"Generated configuration file: {filename}")
                config_counter += 1
    
    print(f"Generated {config_counter} configuration files in {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate random data and configs for NN experiments")
    parser.add_argument('--data-only', action='store_true', help='Generate only the datasets, not configs')
    parser.add_argument('--config-only', action='store_true', help='Generate only the configs, not datasets')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory for generated data')
    parser.add_argument('--config-dir', type=str, default='config/yaml', help='Directory for generated configs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Dataset sizes to generate
    data_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    
    # Generate datasets if needed
    if not args.config_only:
        print(f"Generating {len(data_sizes)} random datasets...")
        datasets = generate_datasets(args.data_dir, data_sizes, args.seed)
        print(f"Generated {len(datasets)} datasets")
    
    # Generate config files if needed
    if not args.data_only:
        print(f"Generating configuration files...")
        generate_config_files(args.config_dir)
