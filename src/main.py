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

from audio_data_generator import (AudioDataGenerator,
                                  generate_audio_data_from_config)
from genetic_algorithm import GeneticAlgorithm, managed_pool
from model import build_and_train_model, build_model
from plotRawData import plot_train_test_with_decision_boundary
from utils import Config  # Add this import; Add these imports
from utils import (ExperimentPaths, cleanup_processes, configure_logging,
                   find_latest_checkpoint, find_latest_results,
                   get_all_child_processes, get_output_dirs,
                   kill_child_processes, load_config, load_results,
                   managed_multiprocessing, plot_results, save_results,
                   terminate_child_processes, validate_audio_file,
                   validate_file, write_config_to_text)

# Add audio imports


# Setup logger
logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info("Received shutdown signal. Cleaning up...")
    cleanup_processes()
    tf.keras.backend.clear_session()
    sys.exit(0)


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
        description='Run audio classification with genetic algorithm.')
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
    parser.add_argument(
        '--audio-files',
        type=str,
        nargs='+',
        required=False,  # Make it optional
        help='Paths to audio files for classification (overrides config file)'
    )
    parser.add_argument(
        '--labels',
        type=int,
        nargs='+',
        required=False,  # Make it optional
        help='Labels for audio files (0 or 1) (overrides config file)'
    )
    return parser.parse_args()


def get_audio_files_and_labels(config, args):
    """Get audio files and labels from config or command line with support for multiple files per class"""
    
    # First try command line arguments
    if args.audio_files and args.labels:
        audio_files = [os.path.expanduser(f) for f in args.audio_files]
        labels = args.labels
        
        if len(audio_files) != len(labels):
            raise ValueError("Number of audio files must match number of labels")
            
        logger.info("Using audio files and labels from command line")
        return audio_files, labels
    
    # Try to get from config file
    if not hasattr(config.audio, 'audio_files') or not config.audio.audio_files:
        raise ValueError("Audio files and labels must be specified either:\n"
                        "  1. Via command line: --audio-files file1.wav file2.wav --labels 0 1\n"
                        "  2. In the config file under audio.audio_files section")
    
    audio_files = []
    labels = []
    
    # Handle config.audio.audio_files as dictionary
    audio_files_config = config.audio.audio_files
    
    # Check if it's a dictionary (which it should be from YAML)
    if isinstance(audio_files_config, dict):
        # New structure: class_0: paths: [file1, file2], class_1: paths: [file3, file4]
        for class_name in sorted(audio_files_config.keys()):
            class_config = audio_files_config[class_name]
            
            # Extract class number from class_name (e.g., 'class_0' -> 0)
            try:
                class_label = int(class_name.split('_')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid class name format: {class_name}. Expected format: class_0, class_1, etc.")
            
            # Handle both 'paths' (list) and 'path' (single file)
            file_paths = []
            if 'paths' in class_config and class_config['paths']:
                file_paths = class_config['paths']
            elif 'path' in class_config and class_config['path']:
                file_paths = [class_config['path']]
            else:
                logger.warning(f"No paths found for {class_name}")
                continue
            
            # Add all files for this class
            for file_path in file_paths:
                expanded_path = os.path.expanduser(file_path)
                if not os.path.exists(expanded_path):
                    raise ValueError(f"Audio file not found: {expanded_path}")
                audio_files.append(expanded_path)
                labels.append(class_label)
    
    # Fallback: check if it's an object with attributes
    elif hasattr(audio_files_config, '__dict__'):
        for class_name in sorted(audio_files_config.__dict__.keys()):
            class_config = getattr(audio_files_config, class_name)
            
            try:
                class_label = int(class_name.split('_')[1])
            except (IndexError, ValueError):
                raise ValueError(f"Invalid class name format: {class_name}. Expected format: class_0, class_1, etc.")
            
            file_paths = []
            if hasattr(class_config, 'paths') and class_config.paths:
                file_paths = class_config.paths
            elif hasattr(class_config, 'path') and class_config.path:
                file_paths = [class_config.path]
            else:
                continue
            
            for file_path in file_paths:
                expanded_path = os.path.expanduser(file_path)
                if not os.path.exists(expanded_path):
                    raise ValueError(f"Audio file not found: {expanded_path}")
                audio_files.append(expanded_path)
                labels.append(class_label)
    
    # Final fallback: old structure with direct lists
    elif hasattr(config.audio, 'audio_files') and hasattr(config.audio, 'labels'):
        if isinstance(config.audio.audio_files, list) and isinstance(config.audio.labels, list):
            audio_files = [os.path.expanduser(f) for f in config.audio.audio_files]
            labels = config.audio.labels
        else:
            raise ValueError("Could not parse audio files from configuration")
    else:
        raise ValueError("Could not parse audio files from configuration")
    
    if not audio_files or not labels:
        raise ValueError("No audio files or labels found in configuration")
    
    if len(audio_files) != len(labels):
        raise ValueError(f"Mismatch: {len(audio_files)} audio files but {len(labels)} labels")
    
    # Validate all files exist
    for file_path in audio_files:
        if not os.path.exists(file_path):
            raise ValueError(f"Audio file not found: {file_path}")
    
    logger.info("Using audio files and labels from configuration file")
    logger.info(f"Found {len(audio_files)} files across {len(set(labels))} classes")
    
    # Log class distribution
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    logger.info(f"Class distribution: {class_counts}")
    
    return audio_files, labels


def optimize_for_cpu():
    """Configure TensorFlow for CPU optimization"""
    # Set number of threads
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(6)

    # Enable MKL if available
    os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '1'

    # Optional: limit memory growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def parse_audio_config(config):
    """Parse audio files from config with support for multiple files per class"""
    if not hasattr(config.audio, 'audio_files'):
        return None, None
    
    audio_files = []
    labels = []
    
    for class_name, class_config in config.audio.audio_files.items():
        # Extract class number from class_name (e.g., 'class_0' -> 0)
        class_label = int(class_name.split('_')[1])
        
        # Handle both single path and multiple paths
        if hasattr(class_config, 'paths') and class_config.paths:
            # Multiple files per class
            for file_path in class_config.paths:
                audio_files.append(os.path.expanduser(file_path))
                labels.append(class_label)
        elif hasattr(class_config, 'path') and class_config.path:
            # Single file per class (backward compatibility)
            audio_files.append(os.path.expanduser(class_config.path))
            labels.append(class_label)
        else:
            logger.warning(f"No paths found for {class_name}")
    
    return audio_files, labels


def main():
    args = parse_arguments()
    config_name = os.path.basename(args.config).replace('.yaml', '')
    paths = ExperimentPaths(config_name)
    configure_logging(args.log, log_path=f"{paths.logs}/experiment.log")

    # Create additional required directories
    os.makedirs('data/preprocessed', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('qc_reports', exist_ok=True)

    logger.debug("Starting main function.")



    
    # Skip if exists logic (same as before)
    # Check if we should skip because files exist
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
        # Load configuration
        config = load_config(args.config)
        logger.info(f"GPU enabled in config: {config.experiment.use_gpu}")

        # Setup compute device
        if not config.experiment.use_gpu:
            optimize_for_cpu()

        # Get audio files and labels (from config or command line)
        try:
            audio_files, labels = get_audio_files_and_labels(config, args)
            logger.info(f"Audio files: {[os.path.basename(f) for f in audio_files]}")
            logger.info(f"Labels: {labels}")
        except ValueError as e:
            logger.error(f"Configuration Error: {e}")
            sys.exit(1)

        # Generate or load audio data with comprehensive QC
        expected_data_file = f"data/raw/{config_name}_data.csv"
        
        if not os.path.exists(expected_data_file):
            logger.info("Generating audio data from files with comprehensive QC...")
            try:
                from audio_data_generator import AudioDataGenerator

                # Convert to class-based structure for the generator
                generator = AudioDataGenerator(config)
    
                class_files_dict = {}
                for file_path, label in zip(audio_files, labels):
                    class_name = f"class_{label}"
                    if class_name not in class_files_dict:
                        class_files_dict[class_name] = []
                    class_files_dict[class_name].append(file_path)
                
                logger.info(f"Organized files: {[(k, len(v)) for k, v in class_files_dict.items()]}")
                
                # Use the new generate_from_files method that expects class_files_dict
                df = generator.generate_from_files(class_files_dict)
                # Save the DataFrame to the expected location
                os.makedirs(os.path.dirname(expected_data_file), exist_ok=True)
                df.to_csv(expected_data_file, index=False)
                logger.info(f"Saved generated data to {expected_data_file}")

            except Exception as e:
                logger.error(f"Error generating audio data: {e}")
                sys.exit(1)
            
        else:
            logger.info(f"Using existing audio data: {expected_data_file}")

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

        # Write configuration parameters to a plain text file
        config_dict = {
            'experiment': vars(config.experiment),
            'data': vars(config.data),
            'audio': vars(config.audio),
            'ga': vars(config.ga),
            'model': vars(config.model),
            'metrics': config.metrics,
            'runtime_info': {
                'audio_files': audio_files,
                'labels': labels,
                'feature_dimensions': len(feature_columns),
                'total_samples': len(df)
            }
        }
        
        config_output_path = f"{paths.config}/parameters.txt"
        os.makedirs(os.path.dirname(config_output_path), exist_ok=True)
        try:
            write_config_to_text(config_dict, config_output_path)
        except Exception as e:
            logger.error(f"Failed to write configuration to text file: {e}")
            sys.exit(1)

        # Split data for audio features
        X = df[[col for col in df.columns if col.startswith('feature_')]].values
        y = df['label'].values
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("TensorFlow GPU support: %s", tf.test.is_built_with_cuda())
        logger.info("TensorFlow GPU available: %s",
                    tf.config.list_physical_devices('GPU'))
       
        # Run genetic algorithm with resume option if specified
        try:
            ga = GeneticAlgorithm(config, X_train, X_test, Y_train, Y_test, df=df, paths=paths)
            
            # Check if we should resume from previous run
            resume_source = None
            if args.resume:
                # First try to find a checkpoint (more complete state)
                checkpoint_path = find_latest_checkpoint(paths)
                if checkpoint_path:
                    logger.info(f"Found checkpoint: {checkpoint_path}")
                    resume_source = checkpoint_path
                else:
                    # Fall back to results file
                    results_path = find_latest_results(paths)
                    if results_path:
                        logger.info(f"Found results file: {results_path}")
                        try:
                            results_data = load_results(results_path)
                            resume_source = results_data
                        except Exception as e:
                            logger.error(f"Failed to load results: {e}")
                            resume_source = None
                    else:
                        logger.info("No previous run found to resume from.")
            
            # Run GA with resume support
            if resume_source:
                logger.info("Resuming from previous run...")
                best_individual, logbook = ga.run(resume_from=resume_source)
            else:
                logger.info("Starting fresh run...")
                best_individual, logbook = ga.run()
            
            # Save final results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_results(best_individual, logbook, paths, timestamp)

            logger.info("Genetic Algorithm completed successfully.")

            # Generate advanced visualizations if enabled
            if config.experiment.generate_advanced_viz:
                logger.info("Generating advanced visualizations...")

                try:
                    from embedding_analysis_plots import (
                        create_embedding_comparison_plots,
                        create_layer_progression_plots,
                        create_classification_curves
                    )
                    from universal_plots import create_statistical_analysis_plots

                    # Build best model for visualization
                    best_model = build_model(config.model)
                    best_model.set_weights(best_individual)

                    # Determine mode (XOR or audio)
                    mode = 'audio' if hasattr(config, 'audio') and config.audio.audio_files else 'xor'

                    # 1. Before/After Network Embedding Comparison
                    logger.info("Creating before/after embedding comparisons...")
                    create_embedding_comparison_plots(
                        best_model, X_train, X_test, Y_train, Y_test,
                        f"{paths.plots}/embedding_comparison.png",
                        mode=mode
                    )

                    # 2. Layer-by-layer progression (if enabled)
                    if config.experiment.viz_layer_progression:
                        logger.info("Creating layer progression plots...")
                        # Use subset of test data for performance
                        sample_size = min(2000, len(X_test))
                        indices = np.random.choice(len(X_test), sample_size, replace=False)
                        create_layer_progression_plots(
                            best_model, X_test[indices], Y_test[indices],
                            f"{paths.plots}/layer_progression.png",
                            max_samples=sample_size
                        )

                    # 3. ROC and Precision-Recall Curves (if enabled)
                    if config.experiment.viz_roc_curves:
                        logger.info("Creating ROC and PR curves...")
                        y_pred_proba = best_model.predict(X_test, verbose=0)
                        metrics = create_classification_curves(
                            Y_test, y_pred_proba,
                            f"{paths.plots}/classification_curves.png"
                        )
                        logger.info(f"Classification metrics: AUC={metrics['roc_auc']:.4f}, AP={metrics['average_precision']:.4f}")

                    # 4. Statistical Analysis (if enabled)
                    if config.experiment.viz_statistical_analysis:
                        logger.info("Creating statistical analysis plots...")
                        y_pred = (y_pred_proba >= 0.5).astype(int).flatten()
                        stats_metrics = create_statistical_analysis_plots(
                            Y_test, y_pred, y_pred_proba.flatten(),
                            f"{paths.plots}/statistical_analysis.png"
                        )
                        logger.info(f"Statistical analysis: Accuracy={stats_metrics['accuracy']:.4f}, "
                                  f"95% CI=[{stats_metrics['ci_lower']:.4f}, {stats_metrics['ci_upper']:.4f}]")

                    # 5. Create universal plots with PCA/t-SNE if enabled
                    logger.info("Creating universal classification plots...")
                    from universal_plots import create_universal_classification_plots
                    create_universal_classification_plots(
                        best_model, X_train, X_test, Y_train, Y_test,
                        f"{paths.plots}/classification_plots_universal.png",
                        title_prefix=f"{mode.upper()} Classification",
                        include_pca=config.experiment.viz_include_pca,
                        include_tsne=config.experiment.viz_include_tsne
                    )

                    # 6. Hyperparameter Impact Analysis (if enabled)
                    if config.experiment.viz_hyperparameter_impact and hasattr(ga, 'plotter'):
                        logger.info("Creating hyperparameter impact analysis...")
                        ga.plotter.create_hyperparameter_impact_plot(
                            logbook, config,
                            f"{paths.plots}/hyperparameter_impact.png"
                        )

                    logger.info("All advanced visualizations completed successfully!")

                except Exception as viz_error:
                    logger.error(f"Error generating advanced visualizations: {viz_error}", exc_info=True)
                    logger.warning("Continuing despite visualization error...")
            
        except Exception as e:
            logger.error(f"Error during Genetic Algorithm execution: {e}", exc_info=True)
            cleanup_processes()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        cleanup_processes()
        sys.exit(1)
    finally:
        # Remove or rename the running indicator file
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


