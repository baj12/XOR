# main.py

import argparse
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm
from model import build_and_train_model
from utils import (load_config, load_results, plot_results, save_results,
                   validate_file)


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
    parser.add_argument(
        '--save', action='store_true', help='Save the GA results after execution.'
    )
    parser.add_argument(
        '--load', type=str, help='Path to load previous GA results.'
    )
    return parser.parse_args()


def main():
    """
    Main function to run the XOR project.

    Steps:
    1. Parse command-line arguments.
    2. Load configuration from YAML file.
    3. Validate the input CSV file.
    4. Split data into training and validation sets.
    5. Initialize and run the Genetic Algorithm.
    6. Display the best individual from the Genetic Algorithm.
    7. Build and train the model using the best individual's weights.
    8. Plot and save the results.
    """
    args = parse_arguments()
    filepath = args.filepath
    config = load_config('config/config.yaml')

    # Access config values
    population_size = config.ga.population_size
    learning_rate = config.model.lr

    if args.load:
        try:
            data = load_results(args.load)
            best_individual = data['best_individual']
            logbook = data['logbook']
            print("Loaded previous GA results successfully.")
        except Exception as e:
            print(f"Error loading previous results: {e}")
            sys.exit(1)
    else:
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
            best_individual, logbook = ga.run()
            print("Genetic Algorithm executed successfully.")
            # best_individual = hall_of_fame[0]
            print("\nBest Individual (Initial Weights):")
            print(best_individual)
            print(f"Fitness: {best_individual.fitness.values}")

        except Exception as e:
            print(
                f"Error during Genetic Algorithm execution: {e}", exc_info=True)
            sys.exit(1)
        # Optionally save results
        if args.save:
            try:
                # best_individual = logbook.select("best")[0]
                print(f"saving results: ")
                save_results(best_individual, logbook)
                print("Results saved successfully.")
            except Exception as e:
                print(f"Error saving results: {e}")

    # Build and train the model using the best individual's weights
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        model = build_and_train_model(best_individual, df, config,
                                      X_train, X_val, y_train, y_val,
                                      model_save_path=f"models/final_model_{current_date}.keras",
                                      plot_accuracy_path=f"plots/final_accuracy_{current_date}.png",
                                      plot_loss_path=f"plots/final_loss_{current_date}.png")
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

    # try:
    #     plot_data_with_decision_boundary(
    #         df_train, df_test, model, save_path='plots/final_evaluation.png')
    # except Exception as e:
    #     print(f"Error during model training: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
