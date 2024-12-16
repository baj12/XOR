import matplotlib.pyplot as plt
import datetime

def plot_results(logbook):
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
