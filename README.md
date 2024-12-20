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

## Detailed Description

### Methods Used

- **Deep Learning (DL):** A neural network is built and trained to solve the XOR problem.
- **Genetic Algorithms (GA):** Used to optimize the initial weights of the neural network.

### Optimization

- **Genetic Algorithm:** Optimizes the initial weights of the neural network to improve training performance.
- **Neural Network Training:** Uses the optimized weights to train the model and achieve better accuracy.

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


