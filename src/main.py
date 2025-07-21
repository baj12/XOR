# main.py

import argparse
import concurrent.futures
import glob
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import signal  # Add this import
import sys
from datetime import datetime

# Add these imports
import librosa
import numpy as np
import psutil
import soundfile as sf
import tensorflow as tf
import yaml
from deap import base, creator, tools
from sklearn.model_selection import train_test_split

# Add audio imports
from audio_data_generator import (AudioDataGenerator,
                                  generate_audio_data_from_config)
from genetic_algorithm import GeneticAlgorithm, managed_pool
from model import build_and_train_model, build_model
from plotRawData import plot_train_test_with_decision_boundary
from utils import (Config,  # Add this import; Add these imports
                   ExperimentPaths, cleanup_processes, configure_logging,
                   find_latest_checkpoint, find_latest_results,
                   get_all_child_processes, get_output_dirs,
                   kill_child_processes, load_config, load_results,
                   managed_multiprocessing, plot_results, save_results,
                   terminate_child_processes, validate_audio_file,
                   validate_file, write_config_to_text)

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
    parser = argparse.ArgumentParser(description='Run audio classification with genetic algorithm.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration YAML file')
    parser.add_argument('--log', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set the logging level')
    parser.add_argument('--skip-if-exists', action='store_true', 
                       help='Skip execution if output PNG files already exist')
    parser.add_argument('--resume', action='store_true', 
                       help='Resume from the last saved state if available')
    parser.add_argument('--audio-files', type=str, nargs='+', required=True,
                       help='Paths to audio files for classification')
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                       help='Labels for audio files (0 or 1)')
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

    if not hasattr(args, 'audio_files') or not args.audio_files:
        logger.error("Audio files not specified. Use --audio-files argument.")
        sys.exit(1)
    
    if not hasattr(args, 'labels') or not args.labels:
        logger.error("Labels not specified. Use --labels argument.")
        sys.exit(1)

    # Create additional required directories
    os.makedirs('data/preprocessed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)

    # # Check if we should skip because files exist
    # if args.skip_if_exists:
    #     # Get the base experiments directory
    #     # This would typically be the parent directory of your paths.plots
    #     experiments_dir = os.path.dirname(os.path.dirname(paths.plots))
        
    #     # # Create search pattern using just the config_name
    #     # search_pattern = f"{experiments_dir}/{config_name}*/plots"
        
    #     # # Search for files
    #     # all_accuracy_files = glob.glob(f"{search_pattern}/accuracy_*.png")
    #     # all_loss_files = glob.glob(f"{search_pattern}/loss_*.png")
        
    #     # if all_accuracy_files and all_loss_files:
    #     #     logger.info(f"Output PNG files for {config_name} already exist and --skip-if-exists is enabled. Skipping execution.")
    #     #     return 
        
    #     # Create search pattern for directories starting with config_name
    #     search_pattern = f"{experiments_dir}/{config_name}*"
        
    #     # Check if any directory matching the pattern exists
    #     matching_dirs = glob.glob(search_pattern)
        
    #     if matching_dirs:
    #         logger.info(f"Directory for {config_name} already exists and --skip-if-exists is enabled. Skipping execution.")
    #         return
    
    # Skip if exists logic (same as before)
    if args.skip_if_exists:
        experiments_dir = os.path.dirname(os.path.dirname(paths.plots))
        search_pattern = f"{experiments_dir}/{config_name}*/plots"
        all_accuracy_files = glob.glob(f"{search_pattern}/accuracy_*.png")
        all_loss_files = glob.glob(f"{search_pattern}/loss_*.png")
        
        if all_accuracy_files and all_loss_files:
            logger.info(f"Output PNG files for {config_name} already exist and --skip-if-exists is enabled. Skipping execution.")
            return
            
        running_files = glob.glob(f"{experiments_dir}/{config_name}*/.running")
        if running_files:
            logger.info(f"Found .running indicator file for {config_name}. Another process is likely working on this configuration. Skipping.")
            return
            
        run_dir = os.path.join(paths.base_dir, '.running')
        with open(run_dir, 'w') as f:
            f.write(f"Started at {datetime.now().isoformat()}")
        logger.debug(f"Created running indicator file at {run_dir}")
    
    try:
        config = load_config(args.config)
        logger.info(f"GPU enabled in config: {config.experiment.use_gpu}")

        if not config.experiment.use_gpu:
            optimize_for_cpu()

        # Generate or load audio data
        expected_data_file = f"data/raw/{config_name}_data.csv"
        
        if not os.path.exists(expected_data_file):
            logger.info("Generating audio data from files...")
            try:
                generate_audio_data_from_config(args.config, args.audio_files, args.labels)
            except Exception as e:
                logger.error(f"Error generating audio data: {e}")
                sys.exit(1)

        # Load and validate data
        try:
            df = validate_audio_file(expected_data_file, config)
            logger.debug(f"Successfully validated the input file: '{expected_data_file}'")
        except ValueError as ve:
            logger.error(f"Validation Error: {ve}")
            sys.exit(1)

        # Update model input dimension based on actual features
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        config.model.input_dim = len(feature_columns)
        logger.info(f"Updated model input dimension to: {config.model.input_dim}")

        # Write configuration
        config_dict = {
            'experiment': vars(config.experiment),
            'data': vars(config.data),
            'audio': vars(config.audio),
            'ga': vars(config.ga),
            'model': vars(config.model),
            'metrics': config.metrics
        }
        
        config_output_path = f"{paths.config}/parameters.txt"
        os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
        write_config_to_text(config_dict, config_output_path)

        # Prepare data for training
        X = df[[col for col in df.columns if col.startswith('feature_')]].values
        y = df['label'].values
        
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("TensorFlow GPU support: %s", tf.test.is_built_with_cuda())
        logger.info("TensorFlow GPU available: %s", tf.config.list_physical_devices('GPU'))

        # Run genetic algorithm
        try:
            ga = GeneticAlgorithm(config, X_train, X_test, Y_train, Y_test, df=df, paths=paths)
            
            if args.resume:
                result_files = glob.glob(f"{paths.results}/ga_results_*.pkl")
                if result_files:
                    latest_result = max(result_files, key=os.path.getmtime)
                    logger.info(f"Resuming from previous run: {latest_result}")
                    try:
                        previous_state = load_results(latest_result)
                        best_individual, logbook = ga.run(resume_from=previous_state)
                    except Exception as e:
                        logger.error(f"Failed to resume from previous state: {e}")
                        logger.info("Starting fresh run instead")
                        best_individual, logbook = ga.run()
                else:
                    logger.info("No previous run found to resume from. Starting fresh run.")
                    best_individual, logbook = ga.run()
            else:
                best_individual, logbook = ga.run()
                
            logger.debug("Genetic Algorithm completed successfully.")
        except Exception as e:
            logger.error(f"Error during Genetic Algorithm execution: {e}", exc_info=True)
            cleanup_processes()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        cleanup_processes()
        sys.exit(1)
    finally:
        run_dir = os.path.join(paths.base_dir, '.running')
        if os.path.exists(run_dir):
            os.remove(run_dir)
            logger.debug("Removed .running indicator file")
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


# Generate audio data and run training
# python main.py \
#    --config config/audio_config.yaml \
#    --audio-files audio/classical.wav audio/rock.wav \
#    --labels 0 1 \
#    --log INFO

