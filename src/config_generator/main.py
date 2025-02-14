import argparse
import os
from itertools import product

import yaml

from .parameter_sets import (ARCHITECTURE_VARIANTS, CLASS_SEPARATION,
                             GA_PARAMS, HIDDEN_LAYERS_RANGE, METRICS_CONFIG,
                             MODEL_PARAMS, NEURONS_RANGE, NOISE_DIMENSIONS)


def get_data_filepath(config_path: str) -> str:
    """
    Get the expected data filepath for a given configuration.
    """
    config_name = os.path.basename(config_path).replace('.yaml', '')
    return os.path.join('data/raw', f"{config_name}_data.csv")


def parse_filename_parameters(filename):
    """Parse parameters from filename."""
    # Example: xor_data_n10000_noise0.50_ratio1.00_seed42.csv
    parts = filename.replace('.csv', '').split('_')
    return {
        'n_samples': int(parts[2][1:]),
        'noise_std': float(parts[3][5:]),
        'ratio_classes': float(parts[4][5:]),
        'random_state': int(parts[5][4:])
    }


def validate_data_parameters(filepath, config):
    """Validate that the data file matches the configuration."""
    # Parse parameters from filename
    filename = os.path.basename(filepath)
    params = parse_filename_parameters(filename)

    # Compare with config
    if (params['noise_std'] != config.experiment.noise_dimensions or
        params['ratio_classes'] != config.data.class_distribution or
            params['n_samples'] != config.data.dataset_size):
        raise ValueError(
            f"Data parameters in file ({params}) do not match "
            f"configuration requirements ({config.data})"
        )


def generate_parameter_combinations():
    """Generate all valid parameter combinations for experiments."""
    experiment_id = 0
    combinations = []

    # Generate combinations
    for noise_dim in NOISE_DIMENSIONS:
        for hidden_layers in HIDDEN_LAYERS_RANGE:
            for neurons in NEURONS_RANGE:
                for sep_type, sep_value in CLASS_SEPARATION.items():
                    for skip_connection in ARCHITECTURE_VARIANTS['skip_connections']:
                        # Skip invalid combinations
                        if hidden_layers == 0 and skip_connection is not None:
                            continue

                        params = {
                            'id': experiment_id,
                            'description': (f"XOR experiment with {noise_dim} noise dimensions, "
                                            f"{hidden_layers} hidden layers"),
                            'hidden_layers': hidden_layers,
                            'neurons': neurons,
                            'noise_dim': noise_dim,
                            'separation': sep_value,
                            'skip_connections': skip_connection,
                            'dataset_size': 10000,
                            # Base XOR (2) + noise dimensions
                            'input_dim': 2 + noise_dim
                        }

                        combinations.append(params)
                        experiment_id += 1

    return combinations


def create_config(params):
    """Create a single configuration dictionary."""
    config = {
        'experiment': {
            'id': str(params['id']),
            'description': params['description'],
            'noise_dimensions': params['noise_dim'],
            'class_separation': params['separation']
        },
        'data': {
            'input_dim': params['input_dim'],  # Correctly set input dimensions
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
            'input_dim': params['input_dim']  # Match data input_dim
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
