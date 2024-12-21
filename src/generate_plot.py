import argparse
import os
import sys

import numpy as np
import pandas as pd

from plotRawData import plot_with_model_file


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        args: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Generate plots for training and testing data with decision boundaries.')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the input CSV data file with columns: x, y, label.')
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to the saved Keras model file (e.g., model.h5).')
    parser.add_argument('--output_file', type=str, default='plots/train_test_decision_boundary.png',
                        help='Path to save the output plot (should end with .png).')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to be used as test set (default: 0.2).')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for data splitting (default: 42).')

    return parser.parse_args()


def validate_file(filepath, file_description):
    """
    Validates the existence of a file.

    Parameters:
        filepath (str): Path to the file.
        file_description (str): Description of the file for error messages.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"{file_description} '{filepath}' does not exist.")


def split_data(df, test_size=0.2, random_state=42):
    """
    Splits the DataFrame into training and testing sets.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'x', 'y', and 'label' columns.
        test_size (float): Fraction of data to be used as test set.
        random_state (int): Random state for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split data as NumPy arrays.
    """
    from sklearn.model_selection import train_test_split

    if not {'x', 'y', 'label'}.issubset(df.columns):
        raise ValueError(
            "Input CSV must contain 'x', 'y', and 'label' columns.")

    X = df[['x', 'y']].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to generate plots based on input data and Keras model.
    """
    args = parse_arguments()

    data_file = args.data_file
    model_file = args.model_file
    output_file = args.output_file
    test_size = args.test_size
    random_state = args.random_state

    # Validate input files
    try:
        validate_file(data_file, "Data file")
        validate_file(model_file, "Model file")
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Read the data
    try:
        df = pd.read_csv(data_file)
        print(f"Data loaded successfully from '{data_file}'.")
    except Exception as e:
        print(f"Error reading data file '{data_file}': {e}")
        sys.exit(1)

    # Split the data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = split_data(
            df, test_size=test_size, random_state=random_state)
        print(
            f"Data split into training and testing sets with test size = {test_size}.")
    except Exception as e:
        print(f"Error splitting data: {e}")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory '{output_dir}'.")

    # Generate the plot
    try:
        plot_with_model_file(
            model_path=model_file,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            save_path=output_file
        )
        print(f"Plot saved successfully as '{output_file}'.")
    except Exception as e:
        print(f"Error generating plot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
