# genetic_algorithm.py

import datetime
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
import multiprocessing
import pandas as pd
import random  # Make sure to import random if using it in eval_individual

# Top-level evaluation function
def eval_individual(individual, df: pd.DataFrame) -> tuple:
    """
    Evaluates the fitness of an individual based on the dataset.

    Parameters:
    - individual (list): A list containing 'x' and 'y' values.
    - df (pd.DataFrame): DataFrame containing the dataset with 'x', 'y', 'label' columns.

    Returns:
    - tuple: A tuple containing the fitness value.
    """
    # Example fitness function: Inverse of distance from origin
    x, y = individual
    distance = (x**2 + y**2)**0.5
    fitness = 1 / (distance + 1e-6)  # Adding a small epsilon to prevent division by zero
    return (fitness,)

def run_genetic_algorithm(df: pd.DataFrame, 
                          population_size: int = 100, 
                          ngen: int = 40, 
                          cxpb: float = 0.5, 
                          mutpb: float = 0.2) -> tuple:
    """
    Runs the genetic algorithm on the provided DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the dataset with 'x', 'y', 'label' columns.
    - population_size (int): Number of individuals in the population.
    - ngen (int): Number of generations to run the algorithm.
    - cxpb (float): Crossover probability.
    - mutpb (float): Mutation probability.

    Returns:
    - population (list): Final population after evolution.
    - logbook (deap.tools.Logbook): Logbook containing statistics per generation.
    - hof (deap.tools.HallOfFame): Hall of Fame containing the best individuals.
    """
    # Define the fitness function: maximize fitness based on evaluation
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator: random floating numbers within the range of 'x' and 'y'
    min_x, max_x = df['x'].min(), df['x'].max()
    min_y, max_y = df['y'].min(), df['y'].max()

    toolbox.register("attr_float", random.uniform, min_x, max_x)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_float, n=2)  # Two attributes: x and y
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function with the DataFrame
    toolbox.register("evaluate", eval_individual, df=df)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population
    population = toolbox.population(n=population_size)

    # Statistics to keep track of
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: sum(f[0] for f in fits) / len(fits))
    stats.register("max", lambda fits: max(f[0] for f in fits))

    # Hall of Fame to store the best individuals
    hof = tools.HallOfFame(1)

    # Use multiprocessing for parallel evaluations
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Run the genetic algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cxpb, mutpb=mutpb,
        ngen=ngen, stats=stats,
        halloffame=hof, verbose=True
    )

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    return population, logbook, hof

def random_float(min_val: float, max_val: float) -> float:
    """
    Generates a random float between min_val and max_val.

    Parameters:
    - min_val (float): Minimum value.
    - max_val (float): Maximum value.

    Returns:
    - float: Random float between min_val and max_val.
    """
    import random
    return random.uniform(min_val, max_val)