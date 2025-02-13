import argparse
import os
from datetime import datetime
from itertools import product

import yaml


def create_config(hidden_layers, neurons_per_layer, ga_params, model_params, output_dir):
    """Create a single configuration dictionary."""
    config = {
        'experiment': {
            'id': str(uuid.uuid4())[:8],
            'description': f"XOR experiment with {hidden_layers} hidden layers",
            'noise_dimensions': 0,
            'class_separation': 0.5
        },
        'data': {
            'input_dim': 2,
            'class_distribution': 0.5,
            'dataset_size': 10000
        },
        'ga': {
            'population_size': ga_params['population_size'],
            'cxpb': ga_params['cxpb'],
            'mutpb': ga_params['mutpb'],
            'ngen': ga_params['ngen'],
            'n_processes': ga_params['n_processes'],
            'max_time_per_ind': ga_params['max_time_per_ind'],
            'epochs': ga_params['epochs']
        },
        'model': {
            'hidden_layers': [neurons_per_layer] * hidden_layers,
            'neurons_per_layer': neurons_per_layer,
            'skip_connections': None,
            'activation': model_params['activation'],
            'optimizer': model_params['optimizer'],
            'lr': model_params['lr'],
            'batch_size': model_params['batch_size'],
            'input_dim': 2
        },
        'metrics': METRICS_CONFIG
    }
    return config


def generate_configs():
    parser = argparse.ArgumentParser(
        description='Generate configuration files for XOR experiments')
    parser.add_argument('--output_dir', type=str, default='config/yaml',
                        help='Directory to store configuration files')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Define parameter ranges
    hidden_layers_range = [0, 1, 2, 3, 4, 5, 10, 20]
    neurons_range = [2, 4, 8, 16, 32, 64]

    # GA parameters
    ga_params = {
        'population_size': [50, 100, 200],
        'crossover_prob': [0.7, 0.8, 0.9],
        'mutation_prob': [0.1, 0.2, 0.3],
        'n_generations': [50, 100, 200],
        'n_processes': [4],  # Adjust based on available CPU cores
        'max_time_per_ind': [300],  # 5 minutes timeout
        'epochs': [10, 20, 50]
    }

    # Model parameters
    model_params = {
        'activation': ['relu', 'tanh'],
        'optimizer': ['adam', 'sgd', 'rmsprop'],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128]
    }

    # Generate combinations
    timestamp = datetime.now().strftime("%Y%m%d")
    config_counter = 0

    for hidden_layers, neurons in product(hidden_layers_range, neurons_range):
        # Create base configuration
        base_config = create_config(
            hidden_layers=hidden_layers,
            neurons_per_layer=neurons,
            ga_params={
                'population_size': 100,
                'cxpb': 0.8,
                'mutpb': 0.2,
                'ngen': 100,
                'n_processes': 4,
                'max_time_per_ind': 300,
                'epochs': 20
            },
            model_params={
                'activation': 'relu',
                'optimizer': 'adam',
                'lr': 0.001,
                'batch_size': 64
            },
            output_dir=args.output_dir
        )

        # Generate filename
        filename = f"config_{timestamp}_{config_counter:03d}_h{hidden_layers}_n{neurons}.yaml"
        filepath = os.path.join(args.output_dir, filename)

        # Save configuration
        with open(filepath, 'w') as f:
            yaml.dump(base_config, f, default_flow_style=False)

        print(f"Generated configuration file: {filename}")
        config_counter += 1


if __name__ == "__main__":
    generate_configs()
