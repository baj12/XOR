import argparse
import logging
import math
import os
import sys

import numpy as np
import pandas as pd

from utils import Config, load_config

# run by
# for fp in config/yaml/config_00*.yaml ; do python -m src.data_generator --config $fp ; done

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


logger = logging.getLogger(__name__)
# Configure basic logging if not configured elsewhere
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s'
)


def generate_xor_data(n_samples=10000, noise_std=0.5, ratio_classes=1.0, noise_dimensions=0, random_state=None):
    """
    Generate XOR dataset with optional noise dimensions.

    Args:
        n_samples (int): Exact number of samples to generate
        noise_std (float): Standard deviation of noise to add to XOR coordinates
        ratio_classes (float): Ratio between classes
        noise_dimensions (int): Number of additional random dimensions to add
        random_state (int): Random seed for reproducibility
    """
    logger.debug(f"Generating XOR data with parameters:")
    logger.debug(f"  n_samples: {n_samples}")
    logger.debug(f"  noise_std: {noise_std}")
    logger.debug(f"  ratio_classes: {ratio_classes}")
    logger.debug(f"  noise_dimensions: {noise_dimensions}")
    logger.debug(f"  random_state: {random_state}")

    if random_state is not None:
        np.random.seed(random_state)

    # Generate exactly n_samples points
    samples_per_quadrant = n_samples // 4
    remainder = n_samples % 4

    # Lists to store data
    X = []
    y = []

    # Generate samples for each quadrant
    for quadrant in range(4):
        # Add extra point to some quadrants if n_samples isn't divisible by 4
        current_samples = samples_per_quadrant + \
            (1 if quadrant < remainder else 0)

        if quadrant == 0:  # Top right (class 1)
            x = np.random.uniform(0, 10, current_samples)
            y_coords = np.random.uniform(0, 10, current_samples)
            label = np.ones(current_samples)
        elif quadrant == 1:  # Bottom left (class 1)
            x = np.random.uniform(-10, 0, current_samples)
            y_coords = np.random.uniform(-10, 0, current_samples)
            label = np.ones(current_samples)
        elif quadrant == 2:  # Top left (class 0)
            x = np.random.uniform(-10, 0, current_samples)
            y_coords = np.random.uniform(0, 10, current_samples)
            label = np.zeros(current_samples)
        else:  # Bottom right (class 0)
            x = np.random.uniform(0, 10, current_samples)
            y_coords = np.random.uniform(-10, 0, current_samples)
            label = np.zeros(current_samples)

        # Add noise if specified
        if noise_std > 0:
            x += np.random.normal(0, noise_std, current_samples)
            y_coords += np.random.normal(0, noise_std, current_samples)

        X.extend(zip(x, y_coords))
        y.extend(label)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Add noise dimensions if specified
    if noise_dimensions > 0:
        noise_features = np.random.normal(0, 1, (n_samples, noise_dimensions))
        X = np.hstack([X, noise_features])

        # Update column names to include noise dimensions
        columns = ['x', 'y'] + \
            [f'noise_{i+1}' for i in range(noise_dimensions)]
    else:
        columns = ['x', 'y']

    # Create DataFrame
    df = pd.DataFrame(X, columns=columns)
    df['label'] = y.astype(int)

    logger.debug(
        f"Generated dataset with {len(df)} samples and {len(df.columns)-1} features")

    return df


def save_data(data, filepath):
    """
    Saves the DataFrame to a CSV file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate XOR Dataset with Noise.")
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Total number of data points to generate (default: 10000)')
    parser.add_argument('--ratio_classes', type=float, default=1.0,
                        help='Ratio between two classes (default: 1.0)')
    parser.add_argument('--noise_std', type=float, default=0.5,
                        help='Standard deviation of Gaussian noise (default: 0.5)')
    parser.add_argument('--output', type=str, default='data/raw/xor_data.csv',
                        help='Output filepath for the generated CSV')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--min_val', type=float, default=-10.0,
                        help='Minimum value for features (default: -10.0)')
    parser.add_argument('--max_val', type=float, default=10.0,
                        help='Maximum value for features (default: 10.0)')
    parser.add_argument('--buffer', type=float, default=0.0,
                        help="Buffer between classes (default: 0.0)")

    return parser.parse_args()

# In data_generator.py


def generate_filename(params):
    """Generate a filename that encodes the data generation parameters."""
    return (f"xor_data_"
            f"n{params['n_samples']}_"
            f"noise{params['noise_std']:.2f}_"
            f"ratio{params['ratio_classes']:.2f}_"
            f"seed{params['random_state']}"
            f".csv")


def parse_filename_parameters(filename):
    """
    Parse parameters from filename.

    Args:
        filename (str): Name of the file

    Returns:
        dict: Dictionary containing parsed parameters
    """
    try:
        # Remove .csv extension and split by underscore
        parts = filename.replace('.csv', '').split('_')
        return {
            'n_samples': int(parts[2][1:]),
            'noise_std': float(parts[3][5:]),
            'ratio_classes': float(parts[4][5:]),
            'random_state': int(parts[5][4:])
        }
    except (IndexError, ValueError) as e:
        raise ValueError(
            f"Invalid filename format: {filename}. Error: {str(e)}")


def generate_data_from_config(config_path: str):
    """Generate data based on YAML configuration file."""
    config = load_config(config_path)

    # Generate data filename based on config filename
    config_name = os.path.basename(config_path).replace('.yaml', '')
    data_filename = f"{config_name}_data.csv"
    data_filepath = os.path.join('data/raw', data_filename)

    # Check if data exists and is valid
    if os.path.exists(data_filepath):
        logger.info(f"Found existing data file: {data_filepath}")
        existing_data = pd.read_csv(data_filepath)

        if validate_existing_data(existing_data, config):
            logger.info("Existing data is valid, skipping generation")
            return data_filepath
        else:
            logger.warning("Existing data is invalid, regenerating")

    # Generate new data using correct parameters from config
    logger.info("Generating new data from configuration")
    data = generate_xor_data(
        n_samples=config.data.dataset_size,
        noise_std=0.5,  # Fixed noise for XOR coordinates
        ratio_classes=config.experiment.class_separation,
        noise_dimensions=config.experiment.noise_dimensions,  # Add noise dimensions
        random_state=42
    )

    # Save data
    os.makedirs(os.path.dirname(data_filepath), exist_ok=True)
    data.to_csv(data_filepath, index=False)
    logger.info(f"Data saved to {data_filepath}")

    return data_filepath


def print_config_values(config: Config):
    """Print relevant configuration values for debugging."""
    logger.debug("Configuration values:")
    logger.debug(f"Dataset size: {config.data.dataset_size}")
    logger.debug(f"Class distribution: {config.data.class_distribution}")
    logger.debug(f"Noise dimensions: {config.experiment.noise_dimensions}")
    logger.debug(f"Class separation: {config.experiment.class_separation}")


def validate_existing_data(data: pd.DataFrame, config: Config) -> bool:
    """
    Validate that existing data matches configuration requirements.

    Args:
        data (pd.DataFrame): Existing data
        config (Config): Configuration object

    Returns:
        bool: True if data is valid, False otherwise
    """
    try:
        # Check number of samples
        if len(data) != config.data.dataset_size:
            logger.warning(
                f"Sample size mismatch: found {len(data)} samples, "
                f"config specifies {config.data.dataset_size}"
            )
            return False

        # Calculate expected number of columns
        expected_columns = ['x', 'y']
        if config.experiment.noise_dimensions > 0:
            expected_columns.extend(
                [f'noise_{i+1}' for i in range(config.experiment.noise_dimensions)])
        expected_columns.append('label')

        # Check columns
        if list(data.columns) != expected_columns:
            logger.warning(
                f"Column mismatch:\n"
                f"Expected columns: {expected_columns}\n"
                f"Found columns: {list(data.columns)}"
            )
            return False

        # Check data types
        if not all(data[col].dtype == np.float64 for col in data.columns if col != 'label'):
            logger.warning("Non-label columns must be float64")
            return False

        if data['label'].dtype != np.int64:
            logger.warning("Label column must be int64")
            return False

        logger.debug("Data validation successful:")
        logger.debug(f"  Samples: {len(data)}")
        logger.debug(f"  Columns: {list(data.columns)}")
        logger.debug(f"  Data types: {data.dtypes}")
        return True

    except Exception as e:
        logger.error(f"Error validating data: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate XOR Dataset from configuration")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--log',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level'
    )
    args = parser.parse_args()

    # Configure logging
    logging_level = getattr(logging, args.log.upper())
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s'
    )

    # Load configuration
    config = load_config(args.config)

    # Print config values for debugging
    print_config_values(config)
    data_filepath = generate_data_from_config(args.config)


if __name__ == "__main__":
    main()
