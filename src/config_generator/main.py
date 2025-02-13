import argparse
import os
from itertools import product

import yaml

from .parameter_sets import (ARCHITECTURE_VARIANTS, CLASS_SEPARATION,
                             GA_PARAMS, HIDDEN_LAYERS_RANGE, METRICS_CONFIG,
                             MODEL_PARAMS, NEURONS_RANGE, NOISE_DIMENSIONS)


def generate_parameter_combinations():
    """Generate all valid parameter combinations for experiments."""
    experiment_id = 0
    combinations = []

    for hidden_layers in HIDDEN_LAYERS_RANGE:
        for neurons in NEURONS_RANGE:
            params = {
                'id': experiment_id,
                'hidden_layers': hidden_layers,
                'neurons': neurons,
                'noise_dim': 0,  # Start with no noise
                'separation': CLASS_SEPARATION['clear'],
                'skip_connections': None,
                'dataset_size': 10000,
                'activation': MODEL_PARAMS['activation']
            }
            combinations.append(params)
            experiment_id += 1

    return combinations


def create_config(params):
    """
    Create a single configuration dictionary.

    Args:
        params (dict): Dictionary containing configuration parameters

    Returns:
        dict: Complete configuration dictionary
    """
    config = {
        'experiment': {
            'id': str(params['id']),
            'description': f"XOR experiment with {params['hidden_layers']} hidden layers",
            'noise_dimensions': params['noise_dim'],
            'class_separation': params['separation']
        },
        'data': {
            'input_dim': 2,
            'class_distribution': params['separation'],
            'dataset_size': params['dataset_size']
        },
        'ga': GA_PARAMS,
        'model': {
            'hidden_layers': [params['neurons']] * params['hidden_layers'],
            'neurons_per_layer': params['neurons'],
            'skip_connections': params['skip_connections'],
            'activation': MODEL_PARAMS['activation'],
            'optimizer': MODEL_PARAMS['optimizer'],
            'lr': MODEL_PARAMS['lr'],
            'batch_size': MODEL_PARAMS['batch_size'],
            'input_dim': 2
        },
        'metrics': METRICS_CONFIG
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Generate configuration files for XOR experiments')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='config/yaml',
        help='Directory to store configuration files'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit the number of configurations to generate'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config_counter = 0
    for params in generate_parameter_combinations():
        if args.limit and config_counter >= args.limit:
            break

        try:
            config = create_config(params)

            filename = f"config_{params['id']:04d}.yaml"
            filepath = os.path.join(args.output_dir, filename)

            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            print(f"Generated configuration file: {filename}")
            config_counter += 1

        except Exception as e:
            print(f"Error generating configuration {params['id']}: {str(e)}")
            continue

    print(
        f"\nGenerated {config_counter} configuration files in {args.output_dir}")


if __name__ == "__main__":
    main()
