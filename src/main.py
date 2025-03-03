# main.py

import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import pickle
import random
import signal  # Add this import
import sys
from datetime import datetime

import numpy as np
import psutil
import tensorflow as tf
import yaml
from deap import base, creator, tools
from sklearn.model_selection import train_test_split

from genetic_algorithm import GeneticAlgorithm, managed_pool
from model import build_and_train_model, build_model
from plotRawData import plot_train_test_with_decision_boundary
from utils import (Config, ExperimentPaths, cleanup_processes,
                   configure_logging, get_all_child_processes, get_output_dirs,
                   kill_child_processes, load_config, load_results,
                   managed_multiprocessing, plot_results, save_results,
                   terminate_child_processes, validate_file,
                   write_config_to_text)

# Setup logger
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal. Cleaning up...")
    cleanup_processes()
    tf.keras.backend.clear_session()
    sys.exit(0)


# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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


def signal_handler(signum, frame):
    logger.info("Received shutdown signal. Cleaning up...")
    cleanup_processes()
    tf.keras.backend.clear_session()
    sys.exit(0)


# Register the signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run XOR neural network training with genetic algorithm.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--log',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level'
    )
    parser.add_argument(
        '--skip-if-exists',
        action='store_true',
        help='Skip execution if output PNG files already exist'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the last saved state if available'
    )
    return parser.parse_args()



def optimize_for_cpu():
    """Configure TensorFlow for CPU optimization"""
    # Set number of threads
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(6)

    # Enable MKL if available
    os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'

    # Optional: limit memory growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def main():
    args = parse_arguments()
    # configure_logging(args.log)
    logger.debug("Starting main function.")
    config_name = os.path.basename(args.config).replace('.yaml', '')
    # Get standardized directory structure
    # dirs = get_output_dirs(config_name)
    # Create paths object
    paths = ExperimentPaths(config_name)
    configure_logging(args.log, log_path=f"{paths.logs}/experiment.log")

    # Check if we should skip because files exist
    if args.skip_if_exists:
        accuracy_files = glob.glob(f"{paths.plots}/accuracy_*.png")
        loss_files = glob.glob(f"{paths.plots}/loss_*.png")
        
        if accuracy_files and loss_files:
            logger.info("Output PNG files already exist and --skip-if-exists is enabled. Skipping execution.")
            return
 
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"GPU enabled in config: {config.experiment.use_gpu}")

        # Setup compute device
        if not config.experiment.use_gpu:  # Change this line
            optimize_for_cpu()
        # Create config dictionary for writing to text file
        config_dict = {
            'experiment': vars(config.experiment),
            'data': vars(config.data),
            'ga': vars(config.ga),
            'model': vars(config.model),
            'metrics': config.metrics
        }
        # Get expected data filepath based on config filename
        config_name = os.path.basename(args.config).replace('.yaml', '')
        expected_data_file = f"data/raw/{config_name}_data.csv"

        # Check if data file exists and matches config
        if not os.path.exists(expected_data_file):
            logger.error(f"Data file not found: {expected_data_file}")
            logger.error("Please generate data first using data_generator.py")
            sys.exit(1)

        # Load and validate data
        try:
            df = validate_file(expected_data_file, config)
            logger.debug(
                f"Successfully validated the input file: '{expected_data_file}'")
        except ValueError as ve:
            logger.error(f"Validation Error: {ve}")
            sys.exit(1)

        # Write configuration parameters to a plain text file
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        # config_output_path = f"plots/config_parameters_{current_date}.txt"
        config_output_path = f"{paths.config}/parameters.txt"
        os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
        try:
            write_config_to_text(config_dict, config_output_path)
        except Exception as e:
            logger.error(f"Failed to write configuration to text file: {e}")
            sys.exit(1)

        # Split data
        X = df[['x', 'y']].values
        y = df['label'].values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        logger.info("TensorFlow GPU support: %s", tf.test.is_built_with_cuda())
        logger.info("TensorFlow GPU available: %s",
                    tf.config.list_physical_devices('GPU'))
        # Run genetic algorithm
        try:
            ga = GeneticAlgorithm(config, X_train, X_test,
                                  Y_train, Y_test, df=df,  paths=paths)
            best_individual, logbook = ga.run()
            logger.debug("Genetic Algorithm completed successfully.")
        except Exception as e:
            logger.error(
                f"Error during Genetic Algorithm execution: {e}", exc_info=True)
            cleanup_processes()
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        cleanup_processes()
        sys.exit(1)
    finally:
        cleanup_processes()


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
        # pool = mp.Pool(initializer=init_worker)
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'pool' in locals():
            pool.close()
            pool.join()
        cleanup_processes()
