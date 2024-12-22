# main.py

import argparse
import logging
import multiprocessing
import os
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm
from model import build_and_train_model
from plotRawData import plot_train_test_with_decision_boundary
from utils import (load_config, load_results, plot_results, save_results,
                   validate_file)

# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG to capture all levels of log messages
#     format='%(asctime)s [%(levelname)s] %(message)s',
#     handlers=[
#         logging.StreamHandler()  # Logs will be printed to the console
#     ]
# )
# logger = logging.getLogger(__name__)
# # Suppress DEBUG messages from matplotlib.font_manager
# logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def configure_logging(log_level=None):
    if not log_level:
        log_level = os.getenv('LOG_LEVEL', 'INFO')

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
    - args (Namespace): Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run XOR Project with Input CSV Data.")
    parser.add_argument(
        'filepath', type=str,
        help='Path to the input CSV file containing x, y, label columns.')
    parser.add_argument(
        '--save', action='store_true', help='Save the GA results after execution.'
    )
    parser.add_argument(
        '--load', type=str, help='Path to load previous GA results.'
    )
    # Define mutually exclusive logging levels
    parser.add_argument(
        '--log',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (DEBUG, INFO, WARNING, ERROR).'
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
    # Configure logging based on the parsed arguments
    configure_logging(args.log)
    logger = logging.getLogger(__name__)

    # Access config values
    population_size = config.ga.population_size
    learning_rate = config.model.lr

    if args.load:
        try:
            data = load_results(args.load)
            best_individual = data['best_individual']
            logbook = data['logbook']
            logger.debug("Loaded previous GA results successfully.")
        except Exception as e:
            logger.error(f"Error loading previous results: {e}")
            sys.exit(1)
    else:
        try:
            # Validate the input file
            df = validate_file(filepath)
            logger.debug(
                f"Successfully validated the input file: '{filepath}'")
        except ValueError as ve:
            logger.error(f"Validation Error: {ve}")
            sys.exit(1)

        # Split data into training and validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(
            df[['x', 'y']].values,
            df['label'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['label'].values
        )

        # Initialize and run Genetic Algorithm
        try:
            ga = GeneticAlgorithm(config, X_train, X_test, Y_train, Y_test)
            best_individual, logbook = ga.run()
            logger.debug("Genetic Algorithm executed successfully.")
            # best_individual = hall_of_fame[0]
            logger.debug("\nBest Individual (Initial Weights):")
            logger.debug(best_individual)
            logger.debug(f"Fitness: {best_individual.fitness.values}")

        except Exception as e:
            logging.error(
                f"Error during Genetic Algorithm execution: {e}",
                exc_info=True)
            sys.exit(1)
        # Optionally save results
        if args.save:
            try:
                # best_individual = logbook.select("best")[0]
                logger.debug(f"saving results: ")

                save_results(best_individual, logbook, filepath='results')
                logger.debug("Results saved successfully.")
            except Exception as e:
                logger.error(f"Error saving results: {e}")

    # Build and train the model using the best individual's weights
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        model = build_and_train_model(best_individual, df, config,
                                      X_train, X_test, Y_train, Y_test,
                                      model_save_path=f"models/final_model_{current_date}.keras",
                                      plot_accuracy_path=f"plots/final_accuracy_{current_date}.png",
                                      plot_loss_path=f"plots/final_loss_{current_date}.png")
        logger.debug("\nModel training completed successfully.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)

    # Plot and save results
    try:
        logger.debug("Begin plot logbook.")
        plot_results(logbook)
        logger.debug("Results plot saved successfully.")
    except Exception as e:
        logger.error(f"Error during plotting results: {e}")

    try:
        plot_train_test_with_decision_boundary(model, X_train, X_test,
                                               Y_train, Y_test,
                                               save_path=f"plots/train_test_decision_boundary_{current_date}.png")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # chatGPT suggests using 'spawn' method for macOS to avoid issues with TensorFlow/Keras
    multiprocessing.set_start_method("spawn")
    main()
