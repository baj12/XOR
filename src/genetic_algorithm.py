import datetime
import matplotlib.pyplot as plt
from deap import algorithms
from deap import base, creator, tools
import multiprocessing


def run_genetic_algorithm():
    # Initialize DEAP toolbox, population, etc.
    # (Add your GA setup code here)

    # Example setup
    toolbox = base.Toolbox()
    # ... define genetic operators

    population = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda x: sum(x)/len(x))
    stats.register("max", max)

    # Run the genetic algorithm
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population, log = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.5, mutpb=0.2,
        ngen=40, stats=stats,
        halloffame=hof, verbose=True
    )

    # Close the pool
    pool.close()
    pool.join()


    return population, log, hof

