# XOR 

## Overview
Solving the XOR problem using deep learning (DL) and genetic algorithms (GA).

The purpose is to better understand how DL works, including neural network optimization using GA, network design, and many other aspects. This is a work in progress.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/baj12/XOR.git
    cd XOR
    ```

2. **Create a virtual environment:**
    *** using conda ***
    ```zsh
    conda env create --file=environment.yml 
    conda activate xorProject
    ```

## Usage

### Generate Data

Generate the XOR dataset using the data generator script:

```sh
python src/data_generator.py
```

### Run the Main Script

Run the main script with the path to the input CSV file containing `x`, `y`, and `label` columns:

```sh
python src/main.py data/raw/xor_data.min1.csv
```

### Clean Directories

Clean and recreate the specified directories:

```sh
python src/cleanDirectories.py
```

### Plot Raw Data

Plot data from CSV files in the `data/raw` directory:

```sh
python src/plotRawData.py
```

### Move Results

Move files from specified directories to a destination directory under `savedResults`:

```sh
python src/moveResults.py <destination_name>
```

### Generate Plot

Generate plots for training and testing data with decision boundaries:

```sh
python src/generate_plot.py --data_file <data_file> --model_file <model_file> --output_file <output_file>
```

## Detailed Description

### Methods Used

- **Deep Learning (DL):** A neural network is built and trained to solve the XOR problem.
- **Genetic Algorithms (GA):** Used to optimize the initial weights of the neural network.

### Optimization

- **Genetic Algorithm:** Optimizes the initial weights of the neural network to improve training performance.
- **Neural Network Training:** Uses the optimized weights to train the model and achieve better accuracy.

### Genetic Algorithm (GA)

The GA works by evolving a population of candidate solutions (individuals) over several generations. Each individual represents a set of initial weights for the neural network. The steps involved are:

1. **Initialization:** Create an initial population of individuals with random weights.
2. **Evaluation:** Evaluate the fitness of each individual by training the neural network with the corresponding weights and measuring its performance.
3. **Selection:** Select individuals based on their fitness to create a mating pool.
4. **Crossover:** Perform crossover (recombination) on pairs of individuals to create offspring.
5. **Mutation:** Apply random mutations to the offspring to introduce variability.
6. **Replacement:** Replace the old population with the new offspring.
7. **Repeat:** Repeat the evaluation, selection, crossover, mutation, and replacement steps for a specified number of generations.

### Neural Network

The neural network used in this project is a simple feedforward neural network with the following structure:

- **Input Layer:** Takes two input features (`x` and `y`).
- **Hidden Layers:** Two hidden layers with configurable units and activation functions.
- **Output Layer:** A single output unit with a sigmoid activation function for binary classification.

The network is trained using the Adam optimizer and binary cross-entropy loss.

### Logging Levels

Logging is configured to provide detailed information during execution. The logging levels can be adjusted in the `model.py` file:

```python
# model.py
logging.basicConfig(
    level=logging.DEBUG,  # Change to logging.INFO or logging.ERROR as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)
```

### Tunable Parameters

Parameters can be tuned in the `config/config.yaml` file:

- **Genetic Algorithm Parameters:**
  - `population_size`: Number of individuals in the population.
  - `cxpb`: Crossover probability.
  - `mutpb`: Mutation probability.
  - `ngen`: Number of generations.
  - `n_processes`: Number of processes for parallel execution.

- **Model Parameters:**
  - `hl1`: Number of units in the first hidden layer.
  - `hl2`: Number of units in the second hidden layer.
  - `activation`: Activation function for hidden layers.
  - `optimizer`: Optimizer to use (e.g., 'adam', 'sgd').
  - `lr`: Learning rate for the optimizer.
  - `batch_size`: Batch size for training.

Example `config/config.yaml`:

```yaml
ga:
  population_size: 50
  cxpb: 0.5
  mutpb: 0.2
  ngen: 100
  n_processes: 4

model:
  hl1: 10
  hl2: 10
  activation: 'relu'
  optimizer: 'adam'
  lr: 0.001
  batch_size: 16
```

### Plotting Results

The results of the genetic algorithm and model training are plotted and saved as images. The plots include:

- Average and Max Fitness over Generations
- Training and Validation Accuracy
- Training and Validation Loss

These plots are saved in the `plots` directory with timestamps.


## things still to do

- allow more flexible network architectures

- allow other input files

- rotate input data to see if this helps to get perfect separations

- use xor problem to evaluate how many generations and individuals one needs depending on complexity of network. i.e. vary the number generations and individuals for a given network complexity, 2,1; 2,2; 16, 16; 32, 32; 64,64; 128,128; 512,512



