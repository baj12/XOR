import argparse
import os
import sys
import pandas as pd
from genetic_algorithm import run_genetic_algorithm
from model import build_and_train_model
from utils import plot_results


def validate_file(filepath):
    """
    Validates that the provided filepath is a CSV file with the required structure:
    - Three columns: 'x', 'y', 'label'
    - 'x' and 'y' are floats
    - 'label' is an integer

    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - df (DataFrame): Validated pandas DataFrame.

    Raises:
    - ValueError: If any validation fails.
    """
    # Check if file exists
    if not os.path.isfile(filepath):
        raise ValueError(f"File '{filepath}' does not exist.")

    # Check file extension
    if not filepath.lower().endswith('.csv'):
        raise ValueError(f"File '{filepath}' is not a CSV file.")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Error reading '{filepath}': {e}")

    # Check number of columns
    expected_columns = ['x', 'y', 'label']
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"CSV file must have exactly three columns: {expected_columns}. Found columns: {list(df.columns)}")

    # Validate data types
    if not pd.api.types.is_float_dtype(df['x']):
        raise ValueError("Column 'x' must contain float values.")
    if not pd.api.types.is_float_dtype(df['y']):
        raise ValueError("Column 'y' must contain float values.")
    if not pd.api.types.is_integer_dtype(df['label']):
        raise ValueError("Column 'label' must contain integer values.")

    return df


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - args (Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run XOR Project with Input CSV Data.")
    parser.add_argument(
        'filepath', type=str, help='Path to the input CSV file containing x, y, label columns.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    filepath = args.filepath

    try:
        # Validate the input file
        df = validate_file(filepath)
        print(f"Successfully validated the input file: '{filepath}'")
    except ValueError as ve:
        print(f"Validation Error: {ve}")
        sys.exit(1)

    # Proceed with Genetic Algorithm
    try:
        final_population, logbook, hall_of_fame = run_genetic_algorithm(df)
        print("\nGenetic Algorithm completed successfully.")
    except Exception as e:
        print(f"Error during Genetic Algorithm execution: {e}")
        sys.exit(1)

    # Display the best individuals
    try:
        print("\nBest Individuals:")
        for ind in hall_of_fame:
            print(ind, ind.fitness.values)
    except Exception as e:
        print(f"Error displaying best individuals: {e}")

    # Build and train the best model
    try:
        build_and_train_model(hall_of_fame, df)
        print("\nModel training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)

    # Plot and save results
    try:
        plot_results(logbook, 'results_plot.png')
        print("Results plot saved as 'results_plot.png'.")
    except Exception as e:
        print(f"Error during plotting results: {e}")


if __name__ == "__main__":
    main()
