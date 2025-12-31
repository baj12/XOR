# XOR & Audio Classification System

## 📚 Quick Navigation

**Choose your path:**

### 🔄 **Continuous Learning (24/7 Audio Classification)**
→ **[START HERE: Beginner-Friendly Guide](README_CONTINUOUS.md)**

For autonomous learning from streaming audio:
- 🎯 **Best for**: Long-term audio experiments, bioacoustics, sound monitoring
- 📧 Automated email reports
- 🧠 Self-improving AI models
- ⚠️ Automatic alerts and drift detection

**Documentation**:
- [📖 Quick Start Guide (Non-Experts)](README_CONTINUOUS.md) ← Start here!
- [⚙️ Implementation Status](docs/CONTINUOUS_LEARNING_STATUS.md)
- [🏗️ Architecture & Strategy](docs/CONTINUOUS_LEARNING_STRATEGY.md)
- [📧 Email Setup Guide](docs/EMAIL_SETUP.md)

---

### 🎯 **XOR Classification (Research & Experiments)**
→ **[XOR Mode Documentation](docs/XOR_MODE_QUICK_START.md)**

For neural network research and genetic algorithm optimization:
- 🔬 **Best for**: Understanding deep learning, GA optimization, research
- 📊 Advanced visualizations (UMAP, t-SNE, decision boundaries)
- 🧬 Genetic algorithm hyperparameter optimization

**Documentation**:
- [📖 XOR Quick Start](docs/XOR_MODE_QUICK_START.md)
- [🎵 Audio Mode Quick Start](docs/AUDIO_MODE_QUICK_START.md)
- [👨‍💻 Developer Guide (CLAUDE.md)](CLAUDE.md)

---

## Overview

This project provides two main capabilities:

1. **Continuous Learning System** (NEW): 24/7 autonomous audio classification from stereo streams
2. **XOR Classification**: Research platform for understanding deep learning and genetic algorithms

Both use genetic algorithms (GA) for neural network optimization and TensorFlow/Keras for training.

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

3. **Setup GPU Monitoring (for Apple Silicon):**
   
   To enable GPU monitoring, you need to configure sudo access for powermetrics:
   ```bash
   # Create a new sudoers file for powermetrics
   sudo visudo -f /etc/sudoers.d/powermetrics
   
   # Add this line (replace yourusername with your username):
   yourusername ALL=(root) NOPASSWD: /usr/bin/powermetrics
   
   # Set proper permissions
   sudo chmod 440 /etc/sudoers.d/powermetrics
   
   # Test the configuration
   sudo powermetrics --show-gpu --show-ane -i 1000 -n 1
   ```

## Usage


### generate config files



### Generate Data

Generate the XOR dataset using the data generator script:

```sh
python src/data_generator.py

for fp in config/yaml/config_*.yaml; do echo $fp ;python src/data_generator.py --config  $fp; done

```

### Run the Main Script

Run the main script with a configuration file:

```sh
python src/main.py --config config/config.yaml --log INFO
```

For cluster usage:
```sh
srun -c 4 --mem 48G -p gpu -q fast --gres=gpu:1 python src/main.py --config config/config.yaml --log DEBUG 2>&1 | tee plots/mylog.xor_data.config.txt
```

### Resource Monitoring

The script automatically monitors:
- CPU usage per process
- Memory usage (Physical and Virtual)
- Thread count
- Process count
- GPU metrics (if on Apple Silicon and properly configured)
  - GPU Utilization
  - GPU Memory usage
  - Neural Engine (ANE) usage

Monitoring data is saved in two files:
- `*_resources.csv`: Contains timestamped metrics in CSV format
- `*_details.log`: Contains detailed per-process information

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

## Configuration

Parameters can be tuned in the `config/config.yaml` file:

- **Genetic Algorithm Parameters:**
  - `population_size`: Number of individuals in the population
  - `cxpb`: Crossover probability
  - `mutpb`: Mutation probability
  - `ngen`: Number of generations
  - `n_processes`: Number of processes for parallel execution
  - `max_time_per_ind`: Maximum time per individual evaluation (in seconds)

- **Model Parameters:**
  - `hl1`: Number of units in the first hidden layer
  - `hl2`: Number of units in the second hidden layer
  - `activation`: Activation function for hidden layers
  - `optimizer`: Optimizer to use (e.g., 'adam', 'sgd')
  - `lr`: Learning rate for the optimizer
  - `batch_size`: Batch size for training

## Future Improvements

- Allow more flexible network architectures
- Allow other input files
- Rotate input data to see if this helps to get perfect separations
- Evaluate GA parameters impact on network complexity
- Add SQL database support for result storage
- Compare CPU vs GPU performance metrics
- Implement Kipoi model descriptions

## Issues

- Monitor and handle non-terminating individuals
- Memory leak investigation
- Performance comparison between CPU and GPU implementations
```
