# main.py

import argparse
import sys

from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm
from model import build_and_train_model
from utils import load_config, plot_results, validate_file


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
    config = load_config('config/config.yaml')
    # Access config values
    population_size = config.ga.population_size
    learning_rate = config.model.lr

    try:
        # Validate the input file
        df = validate_file(filepath)
        print(f"Successfully validated the input file: '{filepath}'")
    except ValueError as ve:
        print(f"Validation Error: {ve}")
        sys.exit(1)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        df[['x', 'y']].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )

    # Initialize and run Genetic Algorithm
    try:
        ga = GeneticAlgorithm(config, X_train, X_val, y_train, y_val)
        pop, log = ga.run()
        print("Genetic Algorithm executed successfully.")
    except Exception as e:
        print(f"Error during Genetic Algorithm execution: {e}")
        sys.exit(1)

    # Display the best individual
    try:
        best_individual = hall_of_fame[0]
        print("\nBest Individual (Initial Weights):")
        print(best_individual)
        print(f"Fitness: {best_individual.fitness.values}")
    except Exception as e:
        print(f"Error displaying best individual: {e}")

    # Build and train the model using the best individual's weights
    try:
        build_and_train_model(best_individual, df)
        print("\nModel training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)

    # Plot and save results
    try:
        plot_results(logbook)
        print("Results plot saved successfully.")
    except Exception as e:
        print(f"Error during plotting results: {e}")


if __name__ == "__main__":
    main()
