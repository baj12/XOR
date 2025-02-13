import argparse
import math
import os

import numpy as np
import pandas as pd


def generate_xor_data(n_samples=10000, noise_std=0.5, ratio_classes=1.0,
                      random_state=None, min_val=-10.0, max_val=10.0, buffer=0):
    """
    Generates a synthetic XOR dataset with noise.

    Parameters:
    - n_samples (int): Total number of samples to generate.
    - noise_std (float): Standard deviation of Gaussian noise to add.
    - ratio_classes (float): Ratio between the classes.
    - random_state (int or None): Seed for reproducibility.
    - min_val (float): Minimum value for features.
    - max_val (float): Maximum value for features.
    - buffer (float): Buffer zone between classes.

    Returns:
    - DataFrame: A pandas DataFrame containing the features and labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Validate parameters
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val.")

    # Number of samples per quadrant
    samples_per_quadrant = n_samples // 4

    # Initialize lists to hold data
    X = []
    y = []
    center = (max_val - min_val)/2 + min_val
    max_val_Min, max_val_Max = sorted([center + buffer, max_val])
    min_val_Min, min_val_Max = sorted([center - buffer, min_val])

    # Generate data for each quadrant
    # Quadrant 1 (upper right) - Label 1
    n_samples_q1 = int(samples_per_quadrant * ratio_classes)
    x1 = np.random.uniform(max_val_Min, max_val_Max, n_samples_q1)
    y1 = np.random.uniform(max_val_Min, max_val_Max, n_samples_q1)
    X.extend(list(zip(x1, y1)))
    y.extend([1] * n_samples_q1)

    # Quadrant 2 (upper left) - Label 0
    n_samples_q2 = int(samples_per_quadrant / ratio_classes)
    x2 = np.random.uniform(min_val_Min, min_val_Max, n_samples_q2)
    y2 = np.random.uniform(max_val_Min, max_val_Max, n_samples_q2)
    X.extend(list(zip(x2, y2)))
    y.extend([0] * n_samples_q2)

    # Quadrant 3 (lower left) - Label 1
    n_samples_q3 = int(samples_per_quadrant * ratio_classes)
    x3 = np.random.uniform(min_val_Min, min_val_Max, n_samples_q3)
    y3 = np.random.uniform(min_val_Min, min_val_Max, n_samples_q3)
    X.extend(list(zip(x3, y3)))
    y.extend([1] * n_samples_q3)

    # Quadrant 4 (lower right) - Label 0
    n_samples_q4 = int(samples_per_quadrant / ratio_classes)
    x4 = np.random.uniform(max_val_Min, max_val_Max, n_samples_q4)
    y4 = np.random.uniform(min_val_Min, min_val_Max, n_samples_q4)
    X.extend(list(zip(x4, y4)))
    y.extend([0] * n_samples_q4)

    # Add noise to the coordinates
    X = np.array(X)
    X += np.random.normal(0, noise_std, X.shape)

    # Create DataFrame
    data = pd.DataFrame(X, columns=['x', 'y'])
    data['label'] = y

    return data


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


def main():
    args = parse_arguments()
    data = generate_xor_data(
        n_samples=args.n_samples,
        ratio_classes=args.ratio_classes,
        noise_std=args.noise_std,
        random_state=args.random_state,
        buffer=args.buffer,
        min_val=args.min_val,
        max_val=args.max_val
    )
    save_data(data, args.output)


if __name__ == "__main__":
    main()
