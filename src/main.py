# main.py

import argparse
import logging
import multiprocessing
import os
import sys
from datetime import datetime

import psutil
from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm, managed_pool
from model import build_and_train_model
from plotRawData import plot_train_test_with_decision_boundary
from utils import (Config, get_all_child_processes, kill_child_processes,
                   load_config, load_results, managed_multiprocessing,
                   plot_results, save_results, terminate_child_processes,
                   validate_file, write_config_to_text)

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
    import logging
    import os
    import sys

    if not log_level:
        log_level = os.getenv('LOG_LEVEL', 'DEBUG')

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {log_level}")
        sys.exit(1)

    # Clear existing handlers to prevent duplicates
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log', mode='a')  # Adds FileHandler
        ]
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully.")

    # Suppress DEBUG logs from specific third-party libraries
    libraries_to_suppress = ['matplotlib']
    for lib in libraries_to_suppress:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logger


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
    parser.add_argument(
        '--config', type=str, default='config/config.yaml', help='Path to config.yaml')
    # Define mutually exclusive logging levels
    parser.add_argument(
        '--log',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (DEBUG, INFO, WARNING, ERROR).'
    )
    pass
    return parser.parse_args()


def cleanup_processes():
    logger.debug("Initiating process cleanup.")

    # Clean up multiprocessing resources
    try:
        multiprocessing.get_context('fork')
        for p in multiprocessing.active_children():
            p.terminate()
            p.join()
    except Exception as e:
        logger.debug(f"Error during multiprocessing cleanup: {e}")

    # Clean up remaining processes
    children = get_all_child_processes()
    if children:
        terminate_child_processes(children)
        psutil.wait_procs(children, timeout=3)

        # Check for remaining processes
        remaining = get_all_child_processes()
        if remaining:
            kill_child_processes(remaining)

    logger.debug("Process cleanup completed.")


def run_experiment(config: Config, data_file: str):
    """Run experiment with given configuration."""
    # Generate data with noise
    data = generate_xor_data(
        n_samples=config.data.dataset_size,
        noise_dim=config.experiment.noise_dimensions,
        class_separation=config.data.class_distribution
    )

    # Create experiment directory
    experiment_dir = f"results/experiment_{config.experiment.id}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment metadata
    metadata = {
        'config': config.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'data_shape': data.shape
    }
    with open(f"{experiment_dir}/metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)

    # Run genetic algorithm
    ga = GeneticAlgorithm(config, X_train, X_val, y_train, y_val)
    best_individual, logbook = ga.run()

    # Save results
    save_results(best_individual, logbook,
                 f"{experiment_dir}/ga_results.pkl")

    # Generate plots
    plot_results(logbook, f"{experiment_dir}/plots")


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
    config_path = args.config

    # Configure logging based on the parsed arguments
    logger = configure_logging(args.log)
    logger.debug("Starting main function.")

    # **Convert Config object to dict**
    try:
        config = load_config(config_path)
        config_dict = {
            'experiment': vars(config.experiment),
            'data': vars(config.data),
            'ga': vars(config.ga),
            'model': vars(config.model),
            'metrics': config.metrics,
            'filepath': filepath,
            'config_path': config_path
        }
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        sys.exit(1)
    # Write configuration parameters to a plain text file
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_output_path = f"plots/config_parameters_{current_date}.txt"
    os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
    try:
        write_config_to_text(config_dict, config_output_path)
    except Exception as e:
        logger.error(f"Failed to write configuration to text file: {e}")
        sys.exit(1)

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

                save_results(best_individual, logbook,
                             filepath=f"final_model_{current_date}.pkl")
                logger.debug("Results saved successfully.")
            except Exception as e:
                logger.error(f"Error saving results - 1: {e}")

    # Build and train the model using the best individual's weights
    try:
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
    logger.debug("All done waiting to exit.")
    for proc in psutil.process_iter(['pid', 'name']):
        if 'python' in proc.info['name']:
            logger.debug(f"Killing process: {proc.info['pid']}")
            proc.kill()

    try:
        logger.debug("All done, starting cleanup.")
        cleanup_processes()

        # Instead of killing the Python process directly, exit cleanly
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        cleanup_processes()
        sys.exit(1)
    finally:
        # Ensure multiprocessing resources are released
        if 'pool' in locals():
            pool.close()
            pool.join()


if __name__ == "__main__":
    # chatGPT suggests using 'spawn' method for macOS to avoid issues with TensorFlow/Keras
    # in the end future seems to be the best option
    with managed_multiprocessing():
        multiprocessing.set_start_method("spawn")
        main()
