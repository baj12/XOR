# src/data_generator.py

import numpy as np
import pandas as pd
import os
import argparse


def generate_xor_data(n_samples=10000, noise_std=0.5, random_state=None):
    """
    Generates a synthetic XOR dataset with noise.

    Parameters:
    - n_samples (int): Total number of samples to generate.
    - noise_std (float): Standard deviation of Gaussian noise to add.
    - random_state (int or None): Seed for reproducibility.

    Returns:
    - DataFrame: A pandas DataFrame containing the features and labels.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Number of samples per quadrant
    samples_per_quadrant = n_samples // 4

    # Initialize lists to hold data
    X = []
    y = []

    # Quadrant I (+,+) and III (-,-) -> Class 0
    for _ in range(samples_per_quadrant):
        x1, x2 = np.random.uniform(0, 10, 2)  # Quadrant I
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(0)

        x1, x2 = np.random.uniform(-10, 0, 2)  # Quadrant III
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(0)

    # Quadrant II (-,+) and IV (+,-) -> Class 1
    for _ in range(samples_per_quadrant):
        x1, x2 = np.random.uniform(-10, 0, 2)  # Quadrant II
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(1)

        x1, x2 = np.random.uniform(0, 10, 2)  # Quadrant IV
        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(1)

    # In case n_samples is not divisible by 4
    remaining = n_samples - len(X)
    for i in range(remaining):
        quadrant = i % 4
        if quadrant == 0:
            x1, x2 = np.random.uniform(0, 10, 2)  # Quadrant I
            label = 0
        elif quadrant == 1:
            x1, x2 = np.random.uniform(-10, 0, 2)  # Quadrant II
            label = 1
        elif quadrant == 2:
            x1, x2 = np.random.uniform(0, 10, 2)  # Quadrant IV
            label = 1
        else:
            x1, x2 = np.random.uniform(-10, 0, 2)  # Quadrant III
            label = 0

        X.append([x1 + np.random.normal(0, noise_std),
                 x2 + np.random.normal(0, noise_std)])
        y.append(label)

    # Create DataFrame
    data = pd.DataFrame(X, columns=['feature1', 'feature2'])
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
    parser.add_argument('--noise_std', type=float, default=0.5,
                        help='Standard deviation of Gaussian noise (default: 0.5)')
    parser.add_argument('--output', type=str, default='data/raw/xor_data.csv',
                        help='Output filepath for the generated CSV (default: data/raw/xor_data.csv)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    return parser.parse_args()


def main():
    args = parse_arguments()
    data = generate_xor_data(
        n_samples=args.n_samples, noise_std=args.noise_std, random_state=args.random_state)
    save_data(data, args.output)


if __name__ == "__main__":
    main()
