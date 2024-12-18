# src/data_generator.py

import numpy as np
import pandas as pd
import os
import argparse
import math


def generate_xor_data(n_samples=10000, ratio_classes=1, noise_std=0.5, random_state=None, min_val=-10.0, max_val=10.0, buffer=0):
    """
    Generates a synthetic XOR dataset with noise.

    Parameters:
    - n_samples (int): Total number of samples to generate.
    - ratio_classes (float): ratio between the classes
    - noise_std (float): Standard deviation of Gaussian noise to add.
    - random_state (int or None): Seed for reproducibility.
    - min_val (float): Minimum value for features. Data will be centered at max - min
    - max_val (float): Maximum value for features.
    - buffer (float): buffer zone between classes

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

    # Quadrant I (+,+) and III (-,-) -> Class 0
    for _ in range(samples_per_quadrant):
        # Quadrant I (+,+)
        x1, x2 = np.random.uniform(max_val_Min, max_val_Max, 2)
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(0)

        # Quadrant III (-,-)
        x1, x2 = np.random.uniform(min_val_Min, min_val_Max, 2)
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(0)

    # Quadrant II (-,+) and IV (+,-) -> Class 1
    for _ in range(math.ceil(samples_per_quadrant * ratio_classes)):
        # Quadrant II (-,+)
        x1 = np.random.uniform(min_val_Min, min_val_Max, 1)[0]
        x2 = np.random.uniform(max_val_Min, max_val_Max, 1)[0]
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(1)

        # Quadrant IV (+,-)
        x2 = np.random.uniform(min_val_Min, min_val_Max, 1)[0]
        x1 = np.random.uniform(max_val_Min, max_val_Max, 1)[0]
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(1)



    # Create DataFrame
    data = pd.DataFrame(X, columns=['x', 'y'])
    data['label'] = y

    return data

def save_data(data, filepath):
    """
    Saves the DataFrame to a CSV file.

    Parameters:
    - data (DataFrame): The data to save.
    - filepath (str): The path where the CSV will be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")



def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - Namespace: Parsed arguments.
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
                        help='Output filepath for the generated CSV (default: data/raw/xor_data.csv)')
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
        n_samples=args.n_samples, ratio_classes=args.ratio_classes, noise_std=args.noise_std, random_state=args.random_state,
        buffer=args.buffer, min_val=args.min_val, max_val=args.max_val)
    save_data(data, args.output)


if __name__ == "__main__":
    main()
