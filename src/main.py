from genetic_algorithm import run_genetic_algorithm
from model import build_and_train_model
from utils import plot_results


def main():
    # Run the Genetic Algorithm
    final_population, logbook, hall_of_fame = run_genetic_algorithm()

    # Display the best individuals
    print("\nBest Individuals:")
    for ind in hall_of_fame:
        print(ind, ind.fitness.values)

    # Build and train the best model
    build_and_train_model(hall_of_fame)

    # Plot and save results
    plot_results(logbook)


if __name__ == "__main__":
    main()
