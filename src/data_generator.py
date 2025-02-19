import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils import Config, load_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_quadrant_data(n_samples: int, quadrant: int,
                           separation: float, noise_dimensions: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data for a specific quadrant with proper separation.

    Args:
        n_samples: Number of samples to generate
        quadrant: Quadrant number (1-4)
        separation: Class separation (0.5, 0.75, or 1.0)
        noise_dimensions: Number of noise dimensions to add

    Returns:
        Tuple of features and labels
    """
    # Calculate range based on separation
    min_val = 1 - separation  # e.g., if separation=0.5, min_val=0.5

    # Generate base coordinates based on quadrant
    if quadrant == 1:  # Top Right, label 1
        x = np.random.uniform(min_val, 1, n_samples)
        y = np.random.uniform(min_val, 1, n_samples)
        label = np.ones(n_samples)
    elif quadrant == 2:  # Top Left, label 0
        x = np.random.uniform(-1, -min_val, n_samples)
        y = np.random.uniform(min_val, 1, n_samples)
        label = np.zeros(n_samples)
    elif quadrant == 3:  # Bottom Left, label 1
        x = np.random.uniform(-1, -min_val, n_samples)
        y = np.random.uniform(-1, -min_val, n_samples)
        label = np.ones(n_samples)
    else:  # Bottom Right, label 0
        x = np.random.uniform(min_val, 1, n_samples)
        y = np.random.uniform(-1, -min_val, n_samples)
        label = np.zeros(n_samples)

    # Add random noise dimensions if specified
    features = np.column_stack([x, y])
    if noise_dimensions > 0:
        noise = np.random.uniform(-1, 1, (n_samples, noise_dimensions))
        features = np.column_stack([features, noise])

    return features, label


def add_overflow_samples(features: np.ndarray, labels: np.ndarray,
                         overflow_ratio: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add overflow samples distributed around all edges, including towards the center.

    Args:
        features: Original feature array
        labels: Original labels
        overflow_ratio: Percentage of samples to overflow (default 2%)
    """
    n_overflow = int(len(features) * overflow_ratio)
    if n_overflow == 0:
        return features, labels

    # Find actual data boundaries for each quadrant
    x_pos = features[features[:, 0] > 0, 0]
    x_neg = features[features[:, 0] < 0, 0]
    y_pos = features[features[:, 1] > 0, 1]
    y_neg = features[features[:, 1] < 0, 1]

    x_pos_min = x_pos.min()
    x_pos_max = x_pos.max()
    x_neg_min = x_neg.min()
    x_neg_max = x_neg.max()
    y_pos_min = y_pos.min()
    y_pos_max = y_pos.max()
    y_neg_min = y_neg.min()
    y_neg_max = y_neg.max()

    extension = 0.1  # 10% extension

    # Helper function to generate overflowed coordinates
    def get_overflow_coord(min_val, max_val, towards_center=False):
        if towards_center:
            if min_val > 0:
                return np.random.uniform(min_val * (1 - extension), min_val)
            else:
                return np.random.uniform(max_val, max_val * (1 - extension))
        else:
            if min_val < 0:
                return np.random.uniform(min_val * (1 + extension), min_val * (1 - extension))
            else:
                return np.random.uniform(max_val * (1 - extension), max_val * (1 + extension))

    overflow_features = []
    overflow_labels = []
    # Now 16 edges (including towards center)
    samples_per_edge = n_overflow // 16

    # Generate overflow samples for each quadrant
    for _ in range(samples_per_edge):
        # Q1 (Top Right, label 1)
        # Outer edges
        x1 = get_overflow_coord(x_pos_min, x_pos_max, False)  # Right edge
        y1 = np.random.uniform(y_pos_min, y_pos_max)
        overflow_features.append([x1, y1])
        overflow_labels.append(1)

        x1 = np.random.uniform(x_pos_min, x_pos_max)
        y1 = get_overflow_coord(y_pos_min, y_pos_max, False)  # Top edge
        overflow_features.append([x1, y1])
        overflow_labels.append(1)

        # Inner edges (towards center)
        x1 = get_overflow_coord(x_pos_min, x_pos_max, True)  # Left edge
        y1 = np.random.uniform(y_pos_min, y_pos_max)
        overflow_features.append([x1, y1])
        overflow_labels.append(1)

        x1 = np.random.uniform(x_pos_min, x_pos_max)
        y1 = get_overflow_coord(y_pos_min, y_pos_max, True)  # Bottom edge
        overflow_features.append([x1, y1])
        overflow_labels.append(1)

        # Q2 (Top Left, label 0)
        # Outer edges
        x2 = get_overflow_coord(x_neg_min, x_neg_max, False)  # Left edge
        y2 = np.random.uniform(y_pos_min, y_pos_max)
        overflow_features.append([x2, y2])
        overflow_labels.append(0)

        x2 = np.random.uniform(x_neg_min, x_neg_max)
        y2 = get_overflow_coord(y_pos_min, y_pos_max, False)  # Top edge
        overflow_features.append([x2, y2])
        overflow_labels.append(0)

        # Inner edges (towards center)
        x2 = get_overflow_coord(x_neg_min, x_neg_max, True)  # Right edge
        y2 = np.random.uniform(y_pos_min, y_pos_max)
        overflow_features.append([x2, y2])
        overflow_labels.append(0)

        x2 = np.random.uniform(x_neg_min, x_neg_max)
        y2 = get_overflow_coord(y_pos_min, y_pos_max, True)  # Bottom edge
        overflow_features.append([x2, y2])
        overflow_labels.append(0)

        # Q3 (Bottom Left, label 1)
        # Outer edges
        x3 = get_overflow_coord(x_neg_min, x_neg_max, False)  # Left edge
        y3 = np.random.uniform(y_neg_min, y_neg_max)
        overflow_features.append([x3, y3])
        overflow_labels.append(1)

        x3 = np.random.uniform(x_neg_min, x_neg_max)
        y3 = get_overflow_coord(y_neg_min, y_neg_max, False)  # Bottom edge
        overflow_features.append([x3, y3])
        overflow_labels.append(1)

        # Inner edges (towards center)
        x3 = get_overflow_coord(x_neg_min, x_neg_max, True)  # Right edge
        y3 = np.random.uniform(y_neg_min, y_neg_max)
        overflow_features.append([x3, y3])
        overflow_labels.append(1)

        x3 = np.random.uniform(x_neg_min, x_neg_max)
        y3 = get_overflow_coord(y_neg_min, y_neg_max, True)  # Top edge
        overflow_features.append([x3, y3])
        overflow_labels.append(1)

        # Q4 (Bottom Right, label 0)
        # Outer edges
        x4 = get_overflow_coord(x_pos_min, x_pos_max, False)  # Right edge
        y4 = np.random.uniform(y_neg_min, y_neg_max)
        overflow_features.append([x4, y4])
        overflow_labels.append(0)

        x4 = np.random.uniform(x_pos_min, x_pos_max)
        y4 = get_overflow_coord(y_neg_min, y_neg_max, False)  # Bottom edge
        overflow_features.append([x4, y4])
        overflow_labels.append(0)

        # Inner edges (towards center)
        x4 = get_overflow_coord(x_pos_min, x_pos_max, True)  # Left edge
        y4 = np.random.uniform(y_neg_min, y_neg_max)
        overflow_features.append([x4, y4])
        overflow_labels.append(0)

        x4 = np.random.uniform(x_pos_min, x_pos_max)
        y4 = get_overflow_coord(y_neg_min, y_neg_max, True)  # Top edge
        overflow_features.append([x4, y4])
        overflow_labels.append(0)

    # Convert to numpy arrays
    overflow_features = np.array(overflow_features)

    # Add noise dimensions if present
    if features.shape[1] > 2:
        noise_dims = features.shape[1] - 2
        noise = np.random.uniform(-1, 1, (len(overflow_features), noise_dims))
        overflow_features = np.hstack([overflow_features, noise])

    # Combine and shuffle
    features_with_overflow = np.vstack([features, overflow_features])
    labels_with_overflow = np.concatenate([labels, overflow_labels])

    shuffle_idx = np.random.permutation(len(features_with_overflow))
    features_with_overflow = features_with_overflow[shuffle_idx]
    labels_with_overflow = labels_with_overflow[shuffle_idx]

    return features_with_overflow, labels_with_overflow


def generate_xor_data(config: Config) -> pd.DataFrame:
    """
    Generate XOR dataset with specified constraints.

    Args:
        config: Configuration object containing:
            - dataset_size: Total number of samples
            - noise_dimensions: Number of noise dimensions
            - class_distribution: Separation between classes (0.5, 0.75, or 1.0)

    Returns:
        DataFrame containing the generated data
    """
    # Validate separation value
    if config.data.class_distribution not in [0.5, 0.75, 1.0]:
        raise ValueError("class_distribution must be one of: 0.5, 0.75, 1.0")

    # Calculate samples per quadrant (ensuring even distribution)
    samples_per_quadrant = config.data.dataset_size // 4

    # Generate data for each quadrant
    features_list = []
    labels_list = []

    for quadrant in range(1, 5):
        features, labels = generate_quadrant_data(
            samples_per_quadrant,
            quadrant,
            config.data.class_distribution,
            config.experiment.noise_dimensions
        )
        features_list.append(features)
        labels_list.append(labels)

    # Combine all quadrants
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    # Add overflow samples
    features, labels = add_overflow_samples(features, labels)

    # Create DataFrame
    columns = ['x', 'y']
    if config.experiment.noise_dimensions > 0:
        columns.extend(
            [f'noise_{i+1}' for i in range(config.experiment.noise_dimensions)])

    df = pd.DataFrame(features, columns=columns)
    df['label'] = labels.astype(int)

    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def generate_data_from_config(config_path: str) -> str:
    """
    Generate data based on YAML configuration file.

    Args:
        config_path: Path to configuration YAML file

    Returns:
        Path to generated data file
    """
    # Load configuration
    config = load_config(config_path)

    # Generate data
    df = generate_xor_data(config)

    # Create output directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)

    # Generate output filename based on config filename
    config_name = os.path.basename(config_path).replace('.yaml', '')
    output_path = f"data/raw/{config_name}_data.csv"

    # Save data
    df.to_csv(output_path, index=False)
    logger.info(f"Generated data saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate XOR dataset")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')

    args = parser.parse_args()
    generate_data_from_config(args.config)
