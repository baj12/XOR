"""
genetic_algorithm.py

because of some bugs in M2/M3 apple silicon and metal which cause at minimum
the macs with M3Pro and tensorflow-metal==1.1.0 to hang at various points.
i had to play with parallelization. in the end concurrent.futures does the trick.
Here we can implement a max wait time for an individual run (individual of the GA)
Of note, we train individual over 10 iterations, which mutation / etc is used?

there is some advanced logging going on that allows logging parallel processes

commented codes allows changing tensorflow number of parallel processes and GPU usage

record fitness was implemented during a search of a memory leak.
What might be better is a link to a SQL data base
"""

"""
Genetic Algorithm Implementation for [Your Project Name]

This script implements a Genetic Algorithm (GA) to optimize [describe what is being optimized, e.g., neural network architectures, parameters, etc.].
The GA utilizes DEAP (Distributed Evolutionary Algorithms in Python) for evolutionary operations.

Classes:
    GeneticAlgorithm: Encapsulates the GA process, including initialization, evaluation, selection, crossover, and mutation.

Functions:
    evaluate_individual(individual, config, X_train, X_val, y_train, y_val):
        Evaluates the fitness of an individual based on model performance.
"""


# from memory_profiler import memory_usage, profile


import contextlib
import gc
import json
import logging
import multiprocessing as mp
import os
import pickle
import random
import signal
import sys
import time
import traceback
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from contextlib import contextmanager
from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from deap import algorithms, base, creator, tools
from pympler import asizeof
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from model import RealTimePlottingCallback, build_model
from plotRawData import plot_train_test_with_decision_boundary
from universal_plots import (create_feature_importance_plot,
                             create_universal_classification_plots)
from utils import Config, get_total_size, save_model_and_history

# TensorFlow configuration
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    # Already initialized - happens in testing
    pass

logger = logging.getLogger(__name__)


# this disables GPU
# tf.config.set_visible_devices([], 'GPU')

class GAProgressPlotter:
    """Real-time plotter for genetic algorithm progress."""
    
    def __init__(self, plot_path, paths):
        self.plot_path = plot_path
        self.paths = paths
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        self.worst_fitness = []
        self.std_fitness = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    
    def update_and_plot(self, generation, population, logbook_record):
        """Update data and create/save plots after each generation."""
        
        # Extract fitness values from population
        fitness_values = [ind.fitness.values[0] for ind in population]
        
        # Store generation data
        self.generations.append(generation)
        self.best_fitness.append(max(fitness_values))
        self.avg_fitness.append(np.mean(fitness_values))
        self.worst_fitness.append(min(fitness_values))
        self.std_fitness.append(np.std(fitness_values))
        
        # Create comprehensive GA progress plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Fitness Evolution
        ax1.plot(self.generations, self.best_fitness, 'g-', label='Best', linewidth=2)
        ax1.plot(self.generations, self.avg_fitness, 'b-', label='Average', linewidth=2)
        ax1.plot(self.generations, self.worst_fitness, 'r-', label='Worst', linewidth=1, alpha=0.7)
        ax1.fill_between(self.generations, 
                        np.array(self.avg_fitness) - np.array(self.std_fitness),
                        np.array(self.avg_fitness) + np.array(self.std_fitness),
                        alpha=0.2, color='blue', label='±1 Std Dev')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('GA Fitness Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Current Population Distribution
        ax2.hist(fitness_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(fitness_values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(fitness_values):.4f}')
        ax2.axvline(max(fitness_values), color='green', linestyle='--', 
                   label=f'Best: {max(fitness_values):.4f}')
        ax2.set_xlabel('Fitness')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Population Fitness Distribution (Gen {generation})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Diversity Over Time (Standard Deviation)
        ax3.plot(self.generations, self.std_fitness, 'purple', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Fitness Standard Deviation')
        ax3.set_title('Population Diversity Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Improvement Rate
        if len(self.best_fitness) > 1:
            improvement = np.diff(self.best_fitness)
            ax4.plot(self.generations[1:], improvement, 'orange', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Fitness Improvement')
            ax4.set_title('Generation-to-Generation Improvement')
        else:
            ax4.text(0.5, 0.5, 'Waiting for data...', 
                    transform=ax4.transAxes, ha='center', va='center')
            ax4.set_title('Generation-to-Generation Improvement')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot (overwrite previous)
        plt.savefig(self.plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save data to CSV for external analysis
        data_path = f"{self.paths.results}/ga_progress.csv"
        progress_df = pd.DataFrame({
            'generation': self.generations,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'worst_fitness': self.worst_fitness,
            'std_fitness': self.std_fitness
        })
        progress_df.to_csv(data_path, index=False)
        
        logger.info(f"Gen {generation}: Best={max(fitness_values):.4f}, "
                   f"Avg={np.mean(fitness_values):.4f}, "
                   f"Std={np.std(fitness_values):.4f}")

    def create_hyperparameter_impact_plot(self, logbook, config, output_path):
        """
        Visualize hyperparameter evolution and impact during GA run.

        Creates a multi-panel visualization showing:
        - Convergence analysis
        - Parameter evolution
        - Performance metrics
        - Resource utilization

        Args:
            logbook: DEAP logbook with generation statistics
            config: Configuration object
            output_path: Path to save the plot
        """
        logger.info("Creating hyperparameter impact visualization...")

        # Extract data from logbook
        gen = [record['gen'] for record in logbook]
        avg_fitness = [record['avg'] for record in logbook]
        max_fitness = [record['max'] for record in logbook]
        min_fitness = [record['min'] for record in logbook]
        std_fitness = [record['std'] for record in logbook]

        # Create figure with 6 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        fig.suptitle('Genetic Algorithm: Hyperparameter Impact Analysis',
                     fontsize=16, fontweight='bold')

        # ===== 1. Convergence Analysis =====
        ax1 = fig.add_subplot(gs[0, :])

        ax1.plot(gen, max_fitness, 'g-', label='Best Fitness', linewidth=2.5, marker='o')
        ax1.plot(gen, avg_fitness, 'b-', label='Average Fitness', linewidth=2, marker='s')
        ax1.fill_between(gen,
                        np.array(avg_fitness) - np.array(std_fitness),
                        np.array(avg_fitness) + np.array(std_fitness),
                        alpha=0.3, color='blue', label='±1 Std Dev')

        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Fitness', fontsize=11)
        ax1.set_title('Convergence Analysis: Fitness Evolution', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Add convergence annotation
        if len(gen) > 1:
            final_improvement = max_fitness[-1] - max_fitness[0]
            ax1.annotate(f'Total Improvement: {final_improvement:+.4f}',
                        xy=(gen[-1], max_fitness[-1]),
                        xytext=(gen[-1]*0.7, max(max_fitness)*0.9),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2),
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # ===== 2. Population Diversity Over Time =====
        ax2 = fig.add_subplot(gs[1, 0])

        ax2.plot(gen, std_fitness, 'purple', linewidth=2.5, marker='d')
        ax2.set_xlabel('Generation', fontsize=11)
        ax2.set_ylabel('Standard Deviation', fontsize=11)
        ax2.set_title('Population Diversity', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add diversity trend line
        if len(gen) > 2:
            z = np.polyfit(gen, std_fitness, 1)
            p = np.poly1d(z)
            ax2.plot(gen, p(gen), "r--", alpha=0.8, label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax2.legend(fontsize=9)

        # ===== 3. Convergence Rate (First Derivative) =====
        ax3 = fig.add_subplot(gs[1, 1])

        if len(gen) > 1:
            convergence_rate = np.diff(max_fitness)
            ax3.plot(gen[1:], convergence_rate, 'orange', linewidth=2, marker='x')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_xlabel('Generation', fontsize=11)
            ax3.set_ylabel('Fitness Change', fontsize=11)
            ax3.set_title('Convergence Rate (∆Fitness per Generation)', fontsize=13, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # Highlight plateau
            plateau_threshold = 0.001
            if len(convergence_rate) > 5:
                recent_changes = convergence_rate[-5:]
                if all(abs(c) < plateau_threshold for c in recent_changes):
                    ax3.text(0.5, 0.95, 'Converged (Plateau Detected)',
                            transform=ax3.transAxes, ha='center', va='top',
                            fontsize=10, color='red', fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        else:
            ax3.text(0.5, 0.5, 'Insufficient data',
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)

        # ===== 4. GA Parameters Summary =====
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')

        params_text = "GA Configuration\n" + "="*40 + "\n\n"
        params_text += f"Population Size:   {config.ga.population_size}\n"
        params_text += f"Generations:       {config.ga.ngen}\n"
        params_text += f"Crossover Prob:    {config.ga.cxpb:.2f}\n"
        params_text += f"Mutation Prob:     {config.ga.mutpb:.2f}\n"
        params_text += f"Epochs/Individual: {config.ga.epochs}\n"
        params_text += f"Parallel Processes: {config.ga.n_processes}\n"
        params_text += f"Max Time/Ind (s):  {config.ga.max_time_per_ind}\n\n"

        params_text += "Model Configuration\n" + "="*40 + "\n\n"
        if hasattr(config.model, 'hidden_layers'):
            params_text += f"Architecture:      {config.model.hidden_layers}\n"
        else:
            params_text += f"Hidden Layers:     {config.model.hl1}, {config.model.hl2}\n"
        params_text += f"Activation:        {config.model.activation}\n"
        params_text += f"Optimizer:         {config.model.optimizer}\n"
        params_text += f"Learning Rate:     {config.model.lr:.6f}\n"
        params_text += f"Batch Size:        {config.model.batch_size}\n"

        ax4.text(0.1, 0.9, params_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # ===== 5. Performance Metrics =====
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        metrics_text = "Performance Metrics\n" + "="*40 + "\n\n"
        metrics_text += f"Initial Best:      {max_fitness[0]:.6f}\n"
        metrics_text += f"Final Best:        {max_fitness[-1]:.6f}\n"
        metrics_text += f"Improvement:       {max_fitness[-1] - max_fitness[0]:+.6f}\n"
        metrics_text += f"Relative Gain:     {((max_fitness[-1]/max_fitness[0])-1)*100:+.2f}%\n\n"

        metrics_text += f"Initial Average:   {avg_fitness[0]:.6f}\n"
        metrics_text += f"Final Average:     {avg_fitness[-1]:.6f}\n"
        metrics_text += f"Avg Improvement:   {avg_fitness[-1] - avg_fitness[0]:+.6f}\n\n"

        metrics_text += f"Initial Diversity: {std_fitness[0]:.6f}\n"
        metrics_text += f"Final Diversity:   {std_fitness[-1]:.6f}\n"
        metrics_text += f"Diversity Change:  {std_fitness[-1] - std_fitness[0]:+.6f}\n\n"

        # Calculate effective generations (where improvement happened)
        if len(gen) > 1:
            improvements = [max_fitness[i] > max_fitness[i-1] for i in range(1, len(max_fitness))]
            effective_gens = sum(improvements)
            metrics_text += f"Effective Gens:    {effective_gens}/{len(gen)-1} ({effective_gens/(len(gen)-1)*100:.1f}%)\n"

        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Hyperparameter impact plot saved to: {output_path}")

        return {
            'initial_best': max_fitness[0],
            'final_best': max_fitness[-1],
            'improvement': max_fitness[-1] - max_fitness[0],
            'convergence_rate': 'converged' if len(gen) > 5 and all(abs(c) < 0.001 for c in np.diff(max_fitness)[-5:]) else 'improving'
        }


def log_gpu_usage():
    """Log GPU usage information."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Get GPU memory info
                memory_info = tf.config.experimental.get_memory_info(gpu.name)
                logger.debug(f"GPU {gpu.name} memory usage: {memory_info}")
        except:
            logger.debug("Could not get GPU memory info")
    else:
        logger.warning("No GPU devices available")


def handler(signum, frame):
    """
    Signal handler to raise a TimeoutError when GA execution exceeds the allowed time.
    """

    location = f"File \"{frame.f_code.co_filename}\", line {frame.f_lineno}, in {frame.f_code.co_name}"
    tb = ''.join(traceback.format_stack(frame))
    error_message = f"GA execution timed out!\nLocation: {location}\nStack Trace:\n{tb}"
    raise TimeoutError(error_message)


# signal a time-out after 1 hour
signal.signal(signal.SIGALRM, handler)
# signal.alarm(3600)


logger = logging.getLogger(__name__)

# Initialize a global counter
fitness_counter = 0


@contextmanager
def managed_pool(max_workers):
    """Improved process pool management with proper resource cleanup"""
    ctx = mp.get_context('spawn')
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        mp_context=ctx
    )

    try:
        yield executor
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for child in mp.active_children():
            try:
                child.terminate()
                child.join(timeout=1.0)
            except Exception as e:
                logger.debug(f"Error cleaning up child process: {e}")
        tf.keras.backend.clear_session()



@contextlib.contextmanager
def timeout_context(seconds):
    def handler(signum, frame):
        raise TimeoutError("Evaluation timed out")
    # Register a function to raise a TimeoutError on the signal
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)


def evaluate_population(self, population, timeout=60):
    """
    Evaluate all individuals in the population with timeout protection.
    """
    pid = os.getpid()
    logger.debug(f"pop created.")

    try:
        with managed_pool(max_workers=self.config.ga.n_processes) as executor:
            # Setup future parallel execution with per individual time-out
            futures = {
                executor.submit(
                    eval_individual,
                    individual=ind,
                    config=self.config,
                    X_train=self.X_train,
                    X_val=self.X_val,
                    y_train=self.y_train,
                    y_val=self.y_val,
                    df=self.df
                ): ind for ind in population
            }
            logger.debug(f"futures created.")

            count = 0
            results = []

            try:
                for future in as_completed(futures, timeout=timeout+1):
                    count += 1
                    # Get the individual associated with this future
                    ind = futures[future]
                    try:
                        fitness = future.result(timeout=timeout)
                        ind.fitness.values = fitness  # Assign fitness to the individual
                        results.append((ind, fitness))
                        logger.debug(f"future. {count}")
                        logger.debug(f"future results {count}: {fitness}")
                    except TimeoutError:
                        logger.warning(
                            f"Evaluation timeout for individual {count}")
                        # Assign worst fitness on timeout
                        ind.fitness.values = (0.0,)
                    except Exception as e:
                        logger.error(
                            f"Error evaluating individual {count}: {e}")
                        # Assign worst fitness on error
                        ind.fitness.values = (0.0,)

            except concurrent.futures.TimeoutError:
                logger.warning(
                    "Global timeout reached during population evaluation")
                # Assign worst fitness to any remaining individuals
                for future, ind in futures.items():
                    if not future.done():
                        ind.fitness.values = (0.0,)

            return population

    except Exception as e:
        logger.error(f"Error during population evaluation: {e}")
        # Ensure all individuals have fitness values
        for ind in population:
            if not hasattr(ind, 'fitness') or not ind.fitness.valid:
                ind.fitness.values = (0.0,)
        return population



def init_worker():
    """Initialize worker process."""
    tf.keras.backend.clear_session()
    # Allow GPU but limit memory growth to prevent memory conflicts
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Optionally, limit GPU memory per process
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]  # 1GB limit
            # )
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")


def is_process_idle(pid, idle_threshold_seconds=15):
    """Check if a process is idle based on CPU usage over time."""
    try:
        process = psutil.Process(pid)
        # Get CPU usage over 1 second interval
        cpu_percent = process.cpu_percent(interval=1.0)
        return cpu_percent < 0.1  # Consider idle if CPU usage is less than 0.1%
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True


def check_tensorflow_hanging(pid):
    """Check if TensorFlow process is hanging by monitoring GPU usage."""
    try:
        process = psutil.Process(pid)
        # Check if process has any threads doing work
        threads = process.threads()
        thread_count = len(threads)

        # If number of threads is abnormally high, might indicate a hang
        if thread_count > 20:
            return True

        # Check memory growth over time
        initial_memory = process.memory_info().rss
        time.sleep(2)
        final_memory = process.memory_info().rss

        # If memory hasn't changed and CPU is idle, might be hanging
        return (final_memory == initial_memory and
                process.cpu_percent() < 0.1)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return True


def init_worker_logging():
    """
    Initializes logging for each worker process.
    """
    import logging
    import sys

    # Configure logging for the worker
    logging.basicConfig(
        # levele should be set by the main process
        # level=logging.INFO,  # Set to DEBUG to capture all debug messages
        format='%(asctime)s [PID %(process)d] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('debug.log', mode='a')
        ]
    )
    # Suppress DEBUG logs from specific third-party libraries if needed
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def monitor_population_progress(futures_dict, stall_threshold=300):
    """
    Monitor overall population evaluation progress.

    Args:
        futures_dict: Dictionary of {future: individual}
        stall_threshold: Time in seconds to wait before considering evaluation stalled
    """
    last_completion_time = time.time()
    last_completion_count = 0

    while futures_dict:
        current_time = time.time()
        current_completion_count = sum(1 for f in futures_dict if f.done())

        # Check if we've made any progress
        if current_completion_count > last_completion_count:
            last_completion_time = current_time
            last_completion_count = current_completion_count
            logger.debug(
                f"Progress: {current_completion_count}/{len(futures_dict)} evaluations complete")

        # Check if we're stalled
        elif current_time - last_completion_time > stall_threshold:
            remaining = len(futures_dict) - current_completion_count
            logger.warning(f"Evaluation stalled: No progress for {stall_threshold}s. "
                           f"{remaining} evaluations remaining")

            # Find the running evaluations
            running_futures = [f for f in futures_dict if not f.done()]
            if running_futures:
                logger.warning("Terminating stalled evaluation batch")
                # Cancel all remaining futures
                for future in running_futures:
                    future.cancel()
                    # Assign worst fitness to the corresponding individuals
                    if future in futures_dict:
                        individual = futures_dict[future]
                        individual.fitness.values = (0.0,)

                return False  # Indicate stall detected

        time.sleep(5)  # Check every 5 seconds

    return True  # All evaluations completed normally


class GeneticAlgorithm:
    """
    Genetic Algorithm class to manage the evolutionary process.

    Attributes:
        config (dict): Configuration parameters from config.yaml.
        X_train (np.ndarray): Training feature data.
        X_val (np.ndarray): Validation feature data.
        y_train (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
        toolbox (deap.Toolbox): DEAP toolbox with registered genetic operators.
        fitness_history (dict): Records fitness statistics per generation.
    """

    def __init__(self, config: Config, X_train, X_val, y_train, y_val, df, paths):
        """
        Initialize the Genetic Algorithm with configuration and data.

        Parameters:
        - config (Config): Configuration object containing GA and model parameters.
        - X_train (np.ndarray): Training features.
        - X_val (np.ndarray): Validation features.
        - y_train (np.ndarray): Training labels.
        - y_val (np.ndarray): Validation labels.
        """
        self.config = config
        self.metrics = {
            'training_time': [],
            'memory_usage': [],
            'convergence': [],
            'parameters': [],
            'capacity_utilization': []
        }
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.df = df
        self.model = build_model(config.model)
        self.total_weights = self.calculate_total_weights()
        self.setup_deap()
        self.pool = mp.Pool(processes=self.config.ga.n_processes)
        self.toolbox.register("map", self.pool.map)
        self.fitness_history = []
        self.counters = defaultdict(int)
        self.pool = None
        self.processes = []
        self.paths = paths
        
        # Resume-related attributes
        self.current_generation = 0
        self.population = None
        self.hall_of_fame = None
        self.logbook = None

    def save_checkpoint(self, generation, population, hof, logbook):
        """Save current state for resume functionality"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{self.paths.results}/checkpoint_gen_{generation}_{timestamp}.pkl"
        
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'hall_of_fame': hof,
            'logbook': logbook,
            'fitness_history': self.fitness_history,
            'config': self.config,
            'total_weights': self.total_weights,
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'timestamp': timestamp
        }
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Also save as latest checkpoint
            latest_checkpoint_path = f"{self.paths.results}/latest_checkpoint.pkl"
            with open(latest_checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            return checkpoint_path
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint for resume"""
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.current_generation = checkpoint_data['generation']
            self.population = checkpoint_data['population']
            self.hall_of_fame = checkpoint_data['hall_of_fame']
            self.logbook = checkpoint_data['logbook']
            self.fitness_history = checkpoint_data.get('fitness_history', [])
            
            # Restore random states
            if 'random_state' in checkpoint_data:
                random.setstate(checkpoint_data['random_state'])
            if 'numpy_random_state' in checkpoint_data:
                np.random.set_state(checkpoint_data['numpy_random_state'])
            
            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Resuming from generation {self.current_generation}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def _evaluate_population(self, population):
        """Evaluate population with timeout protection"""
        timeout = self.config.ga.max_time_per_ind
        
        with managed_pool(max_workers=self.config.ga.n_processes) as executor:
            futures = {
                executor.submit(
                    eval_individual,
                    individual=ind,
                    config=self.config,
                    X_train=self.X_train,
                    X_val=self.X_val,
                    y_train=self.y_train,
                    y_val=self.y_val,
                    df=self.df
                ): ind for ind in population
            }

            count = 0
            for future in as_completed(futures, timeout=timeout + 1):
                count += 1
                individual = futures[future]
                try:
                    fitness = future.result(timeout=timeout)
                    individual.fitness.values = fitness
                    logger.debug(f"Evaluated individual {count}: {fitness}")
                except TimeoutError:
                    logger.warning(f"Evaluation timeout for individual {count}")
                    individual.fitness.values = (0.0,)
                except Exception as e:
                    logger.error(f"Error evaluating individual {count}: {e}")
                    individual.fitness.values = (0.0,)

        return population

    def calculate_total_weights(self) -> int:
        """
        Calculate the total number of weights based on the model architecture.

        Returns:
        - total_weights (int): Total number of weights in the model.
        """
        model = self.model
        total_weights = 0
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                for w in weights:
                    total_weights += w.size
        return total_weights

    def cleanup(self):
        """Cleanup method for GA resources"""
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)

        for process in self.processes:
            try:
                process.terminate()
                process.join(timeout=0.5)
            except Exception as e:
                logger.debug(f"Error cleaning up GA process: {e}")

        tf.keras.backend.clear_session()

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

    def setup_deap(self):
        """
        Set up the DEAP framework for the Genetic Algorithm.
        """
        # Prevent redefining creator classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness,
                           weights=(1.0,))  # Maximize accuracy
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                              self.toolbox.attr_float, n=self.total_weights)
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian,
                              mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def record_fitness(self, population, generation, filename_base='fitness', batch=10):
        logger = logging.getLogger()
        logger.debug(f"Recording fitness started {generation}.")

        pid = os.getpid()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"fitness_history_{timestamp}.json"
        # Increment the counter for the current PID
        self.counters[pid] += 1
        current_count = self.counters[pid]
        logger.debug(
            f"Process {pid}: record_fitness counter = {current_count}.")

        generation_fitness = [ind.fitness.values[0] for ind in population]
        self.fitness_history.append({
            'generation': generation,
            'fitness': generation_fitness
        })
        logger.info(
            f"size of self.fitness_history: {asizeof.asizeof(self.fitness_history)} bytes")
        # Store fitness history to file in batches
        if self.counters[pid] % batch == 0:
            try:
                filepath = os.path.join(self.paths.results, filename)
                with open(filepath, 'a') as f:
                    f.write(f"{self.fitness_history}\n")
                logging.debug(
                    f"Process {pid}: Appended fitness: {self.fitness_history} to {filename}")
                # Clear the fitness_history after saving to file
                self.fitness_history.clear()
                logging.debug(
                    f"Process {pid}: Cleared fitness_history after saving.")

            except Exception as e:
                logging.error(
                    f"Process {pid}: Failed to write to {filename}: {e}")

    def train_best_individual(self, individual):
        """Train the best individual and return model and history"""
        model = build_model(self.config.model)

        # Set weights from individual
        weight_shapes = [w.shape for layer in model.layers
                         for w in layer.get_weights() if w.size > 0]
        weight_tuples = []
        idx = 0
        for shape in weight_shapes:
            size = np.prod(shape)
            weights = np.array(individual[idx:idx+size]).reshape(shape)
            weight_tuples.append(weights)
            idx += size
        model.set_weights(weight_tuples)

        # For audio data, we use the features directly
        X_train_full = self.X_train
        X_val_full = self.X_val
        
        # ADD REAL-TIME PLOTTING HERE
        best_model_plot_path = "plots/current_individual_accuracy.png"
        best_model_loss_path = "plots/current_individual_loss.png"
       
        callbacks = [
            RealTimePlottingCallback(
                plot_path=best_model_plot_path,
                plot_loss_path=best_model_loss_path
            )
        ]

        # Train model
        history = model.fit(
            X_train_full, self.y_train,
            epochs=self.config.ga.epochs,
            batch_size=self.config.model.batch_size,
            validation_data=(X_val_full, self.y_val),
            callbacks=callbacks,
            verbose=1
        )

        return model, history

    def run(self, resume_from=None ):
        """
        Execute the Genetic Algorithm.

        Returns:
        - pop (list): Final population.
        - log (Logbook): Logbook containing statistics of the evolution.
        """
        # Initialize or resume
        if resume_from is not None:
            if isinstance(resume_from, str):
                # It's a file path
                if not self.load_checkpoint(resume_from):
                    logger.warning("Failed to load checkpoint, starting fresh")
                    resume_from = None
            elif isinstance(resume_from, dict):
                # It's checkpoint data
                try:
                    self.current_generation = resume_from['generation']
                    self.population = resume_from['population']
                    self.hall_of_fame = resume_from['hall_of_fame']
                    self.logbook = resume_from['logbook']
                    self.fitness_history = resume_from.get('fitness_history', [])
                    
                    if 'random_state' in resume_from:
                        random.setstate(resume_from['random_state'])
                    if 'numpy_random_state' in resume_from:
                        np.random.set_state(resume_from['numpy_random_state'])
                    
                    logger.info(f"Resuming from generation {self.current_generation}")
                except Exception as e:
                    logger.error(f"Failed to resume from data: {e}")
                    resume_from = None
                # Initialize if not resuming
                
        if resume_from is None:
            self.current_generation = 0
            self.population = self.toolbox.population(n=self.config.ga.population_size)
            self.hall_of_fame = tools.HallOfFame(1)
            self.logbook = tools.Logbook()
            self.logbook.header = ["gen", "avg", "std", "min", "max"]
            
            # Initial population evaluation
            self._evaluate_population(self.population)
            self.hall_of_fame.update(self.population)
            
            # Log initial statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            record = stats.compile(self.population)
            self.logbook.record(gen=0, **record)
            logger.info(f"Generation 0 Statistics: {record}")
            
            # Save initial checkpoint

        # Initialize population and Hall of Fame
        pop = self.toolbox.population(n=self.config.ga.population_size)
        hof = tools.HallOfFame(1)

        # Determine verbose level based on logger level
        log_level = logger.getEffectiveLevel()
        if log_level <= logging.DEBUG:
            verbose = False
        elif log_level <= logging.INFO:
            verbose = False
        else:
            verbose = False

        # stats to keep track of
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        master_logbook = tools.Logbook()
        # Define headers as per your stats
        master_logbook.header = ["gen", "avg", "std", "min", "max"]

        # debugging information
        logger.debug(mp.get_start_method())
        pid = os.getpid()
        timeout = self.config.ga.max_time_per_ind

        # Initialize GA progress plotter
        progress_plot_path = f"{self.paths.plots}/ga_progress_realtime.png"
        ga_plotter = GAProgressPlotter(progress_plot_path, self.paths)
    

        # Parallel environment for evaluating individuals
        with managed_pool(max_workers=self.config.ga.n_processes) as executor:
            logger.info(
                f"Starting Genetic Algorithm execution. Process ID: {pid}")

            # Initialize population
            pop = self.toolbox.population(n=self.config.ga.population_size)
            logger.debug(f"pop created.")

            # Evaluate the entire population
            futures = {executor.submit(eval_individual, ind, self.config,
                                       self.X_train, self.X_val,
                                       self.y_train, self.y_val, self.df): ind for ind in pop}
            logger.debug(f"futures created.")

            count = 0
            killed = 0

            # Initial population evaluation
            # Iterate over futures as they complete
            try:
                for future in as_completed(futures, timeout=timeout+1.0):
                    # Retrieve the individual associated with the future
                    individual = futures[future]

                    count += 1
                    logger.debug(f"future. {count}")
                    f = future.result()
                    logger.debug(f"future results {count}: {f}")
                    individual.fitness.values = f

            except TimeoutError:
                killed += 1
                logger.error(
                    f"Evaluation timed out.{pid}. {count}. {killed}")
            except Exception as e:
                logger.error(
                    f"Error during evaluation for individual {ind}: {e}")
            finally:
                for future in futures:
                    if not future.done():
                        individual = futures[future]
                        individual.fitness.values = (0.0, )
                        future.cancel()
                        logger.warning(
                            f"Total evaluations timed out and were cancelled: {killed}")

        # Optional: Update Hall of Fame if needed
        hof.update(pop)

        # Log statistics
        record = stats.compile(pop)
        master_logbook.record(gen=0, **record)

        logger.debug(
            f"size of hof: {asizeof.asizeof(hof)} bytes")
        logger.debug(
            f"size of self.toolbox: {asizeof.asizeof(self.toolbox)} bytes")

        # Continue evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        start_gen = self.current_generation + 1
        # generations after the parent gen
        for gen in range(start_gen, self.config.ga.ngen + 1):
            logger.info(f"Generation {gen} started.")

            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            logger.debug(f"number of offsprings :{len(offspring)}.")
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            logger.debug(
                f"number of offsprings after map :{len(offspring)}.")

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config.ga.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.config.ga.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            # in case the offspring didn't change we don't need to re-evaluate
            invalid_ind = [
                ind for ind in offspring if not ind.fitness.valid]
            with managed_pool(max_workers=self.config.ga.n_processes) as executor:

                # Evaluate the entire population
                # Setup future parallel execution with per individual time-out
                futures = {executor.submit(eval_individual, ind, self.config,
                                           self.X_train, self.X_val,
                                           self.y_train, self.y_val, self.df): ind for ind in offspring}
                count = 0
                killed = 0
                try:
                    for future in as_completed(futures, timeout=timeout+1):
                        count += 1
                        individual = futures[future]
                        f = future.result(timeout=timeout)
                        individual.fitness.values = f
                        logger.debug(f"Evaluation {pid}. {count} {f}")

                except TimeoutError:
                    killed += 1
                    logger.error(
                        f"Evaluation timed out.{pid}. {count}. {killed}")
                except Exception as e:
                    logger.error(
                        f"Error during evaluation for individual {gen}: {e}")
                finally:
                    cn = 0
                    fn = 0
                    for future in futures:
                        fn += 1
                        if not future.done():
                            cn += 1
                            individual = futures[future]
                            f = (0.0,)
                            logger.debug(f"Evaluation {pid}. {count} {f}")
                            individual.fitness.values = f
                            future.cancel()
                            logger.warning(
                                f"Total evaluations timed out and were cancelled: fn:{fn} - {killed} cn: {cn}")

            # check that not all individuals have 0.0 fitness, which would mean that the evaluation timed out
            if all(ind.fitness.values == (0.0,) for ind in offspring):
                logger.error(
                    "All individuals have 0.0 fitness, indicating that the evaluation timed out.")
                raise RuntimeError(
                    "Evaluation timed out: All individuals have 0.0 fitness.")

            # Replace population with offspring
            pop[:] = offspring
            del offspring

            # Update Hall of Fame
            hof.update(pop)

            # Compile statistics about the new population
            try:
                record = stats.compile(pop)
            except ValueError as e:
                logger.error(f"Error during stats compilation: {e}")
                for i, ind in enumerate(pop):
                    logger.error(
                        f"Individual {i} shape: {np.shape(ind)}, value: {ind}")
                raise e

            # Compile and record statistics
            record = stats.compile(pop)
            master_logbook.record(gen=gen, **record)
            ga_plotter.update_and_plot(gen, pop, record)

            logger.debug(f"Generation {gen} Statistics: {record}")
            self.record_fitness(pop, gen)
            # Record fitness and save checkpoint every 5 generations
            if gen % 5 == 0:
                self.save_checkpoint(gen, self.population, self.hall_of_fame, self.logbook)

            self.current_generation = gen


        # After recording statistics
        for entry in master_logbook:
            logger.debug(entry)
        # current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # fitness_filename = f'fitness_history_{current_date}.json'
        # fitness_filepath = os.path.join("results", fitness_filename)

        # Save fitness history to a JSON file in the results directory
        # with open(fitness_filepath, 'w') as f:
        #    json.dump(self.fitness_history, f)
        # Final checkpoint
        self.save_checkpoint(self.current_generation, self.population, self.hall_of_fame, self.logbook)
        
        # Retrieve the best individual from Hall of Fame
        best_individual = hof[0] if hof else None
        logger.info(
            f"Best Individual: {best_individual}, Fitness: {best_individual.fitness.values if best_individual else 'N/A'}")

        # After evolution completes, train and save the best model
        # best_individual = tools.selBest(population, k=1)[0]
        best_model, best_history = self.train_best_individual(best_individual)

        # Save the best model and its training history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model_and_history(best_model, best_history, self.paths, timestamp)
        
        # Create universal classification plots using UMAP
        try:
            plot_path = f"{self.paths.plots}/universal_classification_{timestamp}.png"
            
            # Determine appropriate title prefix
            if self.X_train.shape[1] == 2:
                title_prefix = "XOR Classification"
            else:
                title_prefix = f"Audio Classification ({self.X_train.shape[1]}D)"
            
            # Create UMAP plots
            X_umap, reducer = create_universal_classification_plots(
                best_model, 
                self.X_train, 
                self.X_val, 
                self.y_train, 
                self.y_val, 
                plot_path,
                title_prefix=title_prefix
            )
            
            # Create feature importance plot
            feature_plot_path = f"{self.paths.plots}/feature_importance_{timestamp}.png"
            create_feature_importance_plot(best_model, feature_plot_path)
            
        except Exception as e:
            logger.error(f"Failed to generate universal classification plots: {e}")

        return best_individual, master_logbook


# @profile
def eval_individual(individual, config, X_train, X_val, y_train, y_val, df):
    """
    Evaluate a single individual.

    Parameters:
        individual: The individual to evaluate
        config: Configuration object
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        df: Full DataFrame containing all features
    """
    try:
        tf.keras.backend.clear_session()

        # Build model with correct input shape
        model = build_model(config.model)

        # Reshape individual to match model weights
        weight_shapes = [w.shape for layer in model.layers
                         for w in layer.get_weights() if w.size > 0]
        weight_tuples = []

        idx = 0
        for shape in weight_shapes:
            size = np.prod(shape)
            weights = np.array(individual[idx:idx+size]).reshape(shape)
            weight_tuples.append(weights)
            idx += size

        model.set_weights(weight_tuples)

        # Prepare data with noise features if specified
        if config.experiment.noise_dimensions > 0:
            noise_cols = [
                f'noise_{i+1}' for i in range(config.experiment.noise_dimensions)]
            noise_data = df[noise_cols].values[: len(X_train)]
            X_train_full = np.column_stack([X_train, noise_data])

            noise_data_val = df[noise_cols].values[len(
                X_train): len(X_train) + len(X_val)]
            X_val_full = np.column_stack([X_val, noise_data_val])

        else:
            X_train_full = X_train
            X_val_full = X_val
        
        # Train and evaluate
        history = model.fit(
            X_train_full, y_train,
            validation_data=(X_val_full, y_val),
            epochs=config.ga.epochs,
            batch_size=config.model.batch_size,
            verbose=0
        )

        val_accuracy = history.history['val_accuracy'][-1]
        return (val_accuracy,)

    except Exception as e:
        logger.error(
            f"Error during individual evaluation: {str(e)}", exc_info=True)
        return (0.0,)


def evaluate_individual(self, individual):
    """Evaluate individual with metric tracking."""
    start_time = time.time()
    memory_start = get_process_memory()

    fitness = super().evaluate_individual(individual)

    if self.config.metrics['tracking']['training']['track_time']:
        self.metrics['training_time'].append(time.time() - start_time)

    if self.config.metrics['tracking']['training']['track_memory']:
        self.metrics['memory_usage'].append(
            get_process_memory() - memory_start)

    return fitness
