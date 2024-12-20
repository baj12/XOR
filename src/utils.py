import datetime
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

import matplotlib.pyplot as plt
import pandas as pd
import yaml


@dataclass
class GAConfig:
    population_size: int
    cxpb: float
    mutpb: float
    ngen: int
    n_processes: int


@dataclass
class ModelConfig:
    hl1: int
    hl2: int
    activation: str
    optimizer: str
    lr: float
    batch_size: int


@dataclass
class Config:
    ga: GAConfig
    model: ModelConfig


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - config (Config): Configuration object containing GA and model parameters.
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    ga_config = GAConfig(**config_dict['ga'])
    model_config = ModelConfig(**config_dict['model'])

    return Config(ga=ga_config, model=model_config)


def plot_results(logbook):
    """
    Plot the results of the genetic algorithm.

    Parameters:
    - logbook (Logbook): Logbook containing statistics of the evolution.
    """
    generations = logbook.select("gen")
    avg_fitness = logbook.select("avg")
    max_fitness = logbook.select("max")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Plot Average and Max Fitness
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label='Average Fitness', color='blue')
    plt.plot(generations, max_fitness, label='Max Fitness', color='red')
    plt.title('Genetic Algorithm Progress Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"plots/xor3_decision_boundary_{timestamp}.png")
    plt.show()


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

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("The CSV file is empty.")

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
        # Print summary information
        print("\nDataset Summary:")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        print("\nColumn info:")
        print(df.dtypes)
        print("\nFirst 5 rows preview:")
        print(df.head())

        return df

    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def save_results(best_individual, logbook, filepath='results/ga_results.pkl'):
    """
    Saves the best individual and logbook to a file using pickle.

    Parameters:
    - best_individual: The best individual from the GA run.
    - logbook: The logbook containing GA run statistics.
    - filepath (str): Path where the results will be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filepath = os.path.join(filepath, f'ga_results_{timestamp}.pkl')

    with open(full_filepath, 'wb') as f:
        pickle.dump(
            {'best_individual': best_individual, 'logbook': logbook}, f)
    print(f"Results saved to {full_filepath}")


def load_results(filepath='results/ga_results.pkl'):
    """
    Loads the best individual and logbook from a file.

    Parameters:
    - filepath (str): Path from where the results will be loaded.

    Returns:
    - data (dict): Dictionary containing 'best_individual' and 'logbook'.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No results found at {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Results loaded from {filepath}")
    return data
