# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project uses deep learning (DL) and genetic algorithms (GA) to solve classification problems. Originally designed for the XOR problem, it has been extended to support **audio classification** from WAV files. The system uses DEAP for genetic algorithm optimization and TensorFlow/Keras for neural network training.

**NEW**: The project now includes a **continuous learning system** for autonomous 24/7 audio classification. See [Continuous Learning Status](docs/CONTINUOUS_LEARNING_STATUS.md) and [Strategy](docs/CONTINUOUS_LEARNING_STRATEGY.md).

## Critical Distinction: Input Data Types

The codebase handles **two fundamentally different input types**:

### 1. XOR Data (Synthetic)
- Generated via `data_generator.py`
- Creates 2D or multi-dimensional synthetic data with configurable class separation
- Features: x, y coordinates plus optional noise dimensions
- Used for testing neural network architectures

### 2. Audio Data (WAV files)
- Processed via `audio_data_generator.py`
- Extracts features from audio files (MFCC, spectral, chroma, tonnetz)
- **Special handling for rubix44**: Large multi-hour WAV recordings in `data/rubix44/`
- Multiple files per class are supported
- Generates comprehensive QC reports with spectrograms, waveform analysis, and compatibility checks

**When working with WAV files, always use `audio_data_generator.py`, not `data_generator.py`.**

## Common Commands

### Environment Setup
```bash
# Using conda (recommended)
conda env create --file=environment.macM.yml  # Mac Apple Silicon
# OR
conda env create --file=environment.Unix.yml  # Linux/Unix
conda activate xorProject
```

### Generate Data

**For XOR (synthetic) data:**
```bash
python src/data_generator.py --config config/yaml/config_XXXX.yaml
```

**For audio (WAV) data:**
```bash
# Data generation is handled automatically by main.py if files don't exist
# Or run explicitly:
python src/audio_data_generator.py --config config/audio_config.yaml \
    --audio-files path1.wav path2.wav --labels 0 1
```

### Run Training

**Standard run:**
```bash
python src/main.py --config config/yaml/config_0518.yaml --log INFO
```

**Skip if results exist:**
```bash
python src/main.py --config config/yaml/config_0518.yaml --skip-if-exists
```

**Resume from checkpoint:**
```bash
python src/main.py --config config/yaml/config_0518.yaml --resume
```

**With audio files (override config):**
```bash
python src/main.py --config config/audio_config.yaml \
    --audio-files data/rubix44/file1.wav data/rubix44/file2.wav \
    --labels 0 1 --log INFO
```

**Cluster usage:**
```bash
srun -c 4 --mem 48G -p gpu -q fast --gres=gpu:1 \
    python src/main.py --config config/config.yaml --log DEBUG
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_genetic_algorithm.py -v
```

### Utilities
```bash
# Clean directories
python src/cleanDirectories.py

# Plot raw data
python src/plotRawData.py

# Move results to savedResults
python src/moveResults.py <destination_name>

# Generate plots for specific model
python src/generate_plot.py --data_file <data> --model_file <model> --output_file <output>
```

## Architecture Overview

### Data Flow

```
Config YAML → Data Generator → Features/Labels → Train/Test Split → GA Optimization → Best Model
                   ↓                                                        ↓
              (XOR or Audio)                                         Real-time plots
```

### Key Components

#### 1. Configuration System (`utils.py`)
- `Config`: Master configuration dataclass
- `ExperimentConfig`: Experiment metadata and GPU settings
- `DataConfig`: Dataset parameters (size, separation, input dimensions)
- `AudioConfig`: Audio processing parameters (sample rate, features, file paths)
- `GAConfig`: Genetic algorithm hyperparameters
- `ModelConfig`: Neural network architecture
- `ExperimentPaths`: Manages directory structure for outputs

**Important**: Audio files can be specified in config under `audio.audio_files` with structure:
```yaml
audio:
  audio_files:
    class_0:
      paths: [file1.wav, file2.wav]  # Multiple files per class
    class_1:
      paths: [file3.wav, file4.wav]
```

#### 2. Data Generation

**`data_generator.py` (XOR/Synthetic)**
- Generates quadrant-based XOR data with configurable class separation (0.5, 0.75, 1.0)
- Supports noise dimensions
- Adds overflow samples at boundaries

**`audio_data_generator.py` (WAV files)**
- `AudioFileAnalyzer`: Analyzes WAV files for compatibility (sample rate, channels, duration)
- `AudioProcessor`: Extracts features from audio segments (MFCC, spectral, chroma, tonnetz)
- `AudioDataGenerator`: Main orchestrator with comprehensive QC
- **Memory-optimized** for large files (processes in chunks)
- Generates QC reports in `qc_reports/` with:
  - Audio summaries and statistics
  - Spectrograms and waveforms
  - Feature comparison radar charts
  - Compatibility assessments

#### 3. Model Building (`model.py`)
- `build_model()`: Creates neural networks with skip connections (residual, dense, or none)
- `RealTimePlottingCallback`: Plots accuracy/loss after each epoch
- Supports Adam, SGD, RMSprop optimizers
- Configurable hidden layers and activation functions

#### 4. Genetic Algorithm (`genetic_algorithm.py`)
- `GeneticAlgorithm`: Main GA class using DEAP
- Optimizes model hyperparameters (learning rate, batch size, layer sizes)
- **Parallel evaluation** using `ProcessPoolExecutor` (fixes Apple Silicon Metal hanging issues)
- **Timeout handling** for individual evaluations (`max_time_per_ind`)
- `GAProgressPlotter`: Real-time visualization of GA convergence
- Checkpoint/resume support for long runs

**Note on Apple Silicon**: Parallelization uses `concurrent.futures` due to TensorFlow Metal bugs causing hangs with standard multiprocessing.

#### 5. Main Execution (`main.py`)
- Orchestrates entire pipeline
- **Automatic data type detection**: Uses `audio_data_generator.py` for WAV files, `data_generator.py` for XOR
- Handles both command-line audio file arguments and config-based file lists
- Creates `.running` indicator files to prevent duplicate runs with `--skip-if-exists`
- Supports resume from checkpoints or results files
- Comprehensive logging and resource monitoring

### Directory Structure

```
XOR/
├── config/yaml/          # YAML config files (auto-generated, 1000+ configs)
├── data/
│   ├── raw/             # Generated datasets (CSV)
│   ├── preprocessed/    # Preprocessed audio features (NPZ)
│   ├── rubix44/         # Large WAV files for audio experiments
│   └── backgroundNoise/ # Additional audio samples
├── experiments_<timestamp>_<config>/  # Output per experiment
│   ├── plots/           # Training plots, decision boundaries
│   ├── logs/            # Execution logs
│   ├── models/          # Saved Keras models
│   ├── checkpoints/     # GA checkpoints (pickle)
│   └── config/          # Runtime config snapshots
├── qc_reports/          # Audio QC reports with spectrograms
├── src/
│   ├── main.py                    # Main entry point
│   ├── data_generator.py          # XOR data generation
│   ├── audio_data_generator.py    # Audio data generation + QC
│   ├── genetic_algorithm.py       # GA implementation
│   ├── model.py                   # Neural network builder
│   ├── utils.py                   # Config, paths, helpers
│   ├── universal_plots.py         # UMAP/PCA/t-SNE classification plots
│   ├── embedding_analysis_plots.py # Before/after network, ROC, layer progression
│   └── config_generator/          # Config generation system
└── tests/               # pytest test suite
```

## Configuration Files

Config files are stored in `config/yaml/` and follow this structure:

```yaml
experiment:
  id: "518"
  description: "XOR experiment with 1 noise dimensions, 4 hidden layers"
  noise_dimensions: 1
  use_gpu: true

data:
  dataset_size: 10000
  class_distribution: 0.75  # Class separation: 0.5, 0.75, or 1.0
  input_dim: 3

audio:
  sample_rate: 22050
  segment_duration: 1.0
  n_mfcc: 13
  feature_types: ['mfcc', 'spectral', 'chroma']
  audio_files:
    class_0:
      paths: ['~/data/rubix44/exp5.empty.noise.wav']
    class_1:
      paths: ['~/data/rubix44/lavendar.1.noise.2hrs.wav']

ga:
  population_size: 100
  cxpb: 0.8          # Crossover probability
  mutpb: 0.2         # Mutation probability
  ngen: 10           # Generations
  epochs: 10         # Training epochs per individual
  n_processes: 1     # Parallel processes
  max_time_per_ind: 72000  # Timeout per individual (seconds)

model:
  hidden_layers: [64, 64, 64, 64]
  activation: relu
  optimizer: adam
  lr: 0.001
  batch_size: 32
  skip_connections: dense  # Options: null, 'residual', 'dense'
```

## GPU and Performance

### GPU Configuration
- Set `experiment.use_gpu: true` in config to enable GPU
- On Apple Silicon, monitor with: `sudo powermetrics --show-gpu --show-ane -i 1000 -n 1`
- CPU optimization is automatic when GPU disabled (uses MKL, multi-threading)

### Memory Management
- Audio processing uses chunked loading for large files
- TensorFlow session clearing after each GA individual
- Garbage collection forced at strategic points
- `max_time_per_ind` prevents runaway evaluations

### Parallel Processing
- Set `ga.n_processes` in config (typically 1 for GPU, 4-8 for CPU)
- Uses `concurrent.futures.ProcessPoolExecutor` for stability on Apple Silicon

## Working with Audio (rubix44)

The `data/rubix44/` directory contains large multi-hour WAV recordings. Key considerations:

1. **File Size**: Files are 1-2.7 GB each; processing is memory-optimized with chunked loading
2. **Multiple Files per Class**: Config supports multiple WAV files per class label
3. **QC Reports**: Always check `qc_reports/` after generation for:
   - Sample rate compatibility
   - Dynamic range comparison
   - Spectral feature distribution
   - Class balance

4. **Feature Extraction**:
   - Default segment duration: 1 second
   - Samples per file: 1000 (configurable via `data.samples_per_file`)
   - Features normalized with StandardScaler

5. **Example Config for rubix44**:
```yaml
audio:
  audio_files:
    class_0:
      paths:
        - '~/data/rubix44/exp5.empty.noise.wav'
        - '~/data/rubix44/exp6 - 1 rubix 1empty.noise.wav'
    class_1:
      paths:
        - '~/data/rubix44/lavendar.1.noise.2hrs.wav'
        - '~/data/rubix44/lavendel in aufnahmebeaker noise part - 1 rubix 1.wav'
```

## Logging and Monitoring

- Set log level via `--log` flag: DEBUG, INFO, WARNING, ERROR
- Logs saved to `experiments_*/logs/experiment.log`
- Resource monitoring automatic (CPU, memory, GPU if available)
- Real-time plots update during training
- GA progress plots show convergence over generations

## Execution Skipping Logic

The `--skip-if-exists` flag prevents re-running completed experiments:
- Checks for `accuracy_*.png` and `loss_*.png` in any matching experiment directory
- Creates `.running` indicator file during execution
- Skips if another process is already running (detected via `.running` file)
- Removes `.running` file on completion or failure

## Resume Capability

Use `--resume` to continue interrupted runs:
- First tries to load from checkpoint (most recent in `checkpoints/`)
- Falls back to results file if no checkpoint
- Preserves GA population and logbook state
- Useful for long-running experiments on clusters with time limits

## User Documentation

For complete guides on using this system, see:

- **[XOR Mode Quick Start](docs/XOR_MODE_QUICK_START.md)**: Complete guide for XOR classification including configuration, data generation, visualization interpretation, and troubleshooting
- **[Audio Mode Quick Start](docs/AUDIO_MODE_QUICK_START.md)**: Complete guide for audio/WAV file classification including rubix44 dataset usage, feature extraction, QC reports, and audio-specific troubleshooting
- **[Continuous Learning Status](docs/CONTINUOUS_LEARNING_STATUS.md)**: Implementation status, gaps, and production readiness for the continuous learning system
- **[Continuous Learning Strategy](docs/CONTINUOUS_LEARNING_STRATEGY.md)**: Architecture and design document for autonomous 24/7 audio classification

These guides cover:

- Quick start commands
- Configuration structure and all parameters
- Understanding input data and transformations
- Interpreting all visualizations (embedding comparisons, layer progression, ROC curves, decision boundaries, statistical analysis)
- Common issues and solutions
- Performance optimization
- Advanced usage patterns

## Continuous Learning System

The `src/continuous/` module provides autonomous 24/7 audio classification from stereo WAV streams.

### Key Features

- **Stereo Processing**: Left channel = positive class, Right channel = negative class
- **Incremental Training**: Sliding window learning (configurable weeks)
- **Drift Detection**: Monitors feature distribution changes over time
- **Email Alerts**: Performance drops, drift, storage issues
- **Weekly Reports**: Comprehensive performance summaries
- **Autonomous Operation**: Runs indefinitely with scheduled training
- **Rubix44 Integration**: Automatic polling and processing of recordings from rubix44-recorder API

### Data Providers

The system supports two data ingestion methods:

#### 1. Rubix44 API Provider (Recommended for Production)

Polls the [rubix44-recorder](https://github.com/baj12/rubix44-recorder) API for new stereo recordings and automatically processes them.

**Configuration** (config/continuous_learning_config.yaml):
```yaml
orchestration:
  data_provider: rubix44  # Use rubix44 API

  rubix44:
    api_url: http://10.0.0.58:5000  # URL of rubix44-recorder API
    poll_interval_minutes: 5  # Check every 5 minutes
    download_dir: data/continuous/recordings  # Local download directory
    output_prefix_filter: null  # Filter by prefix (null = all files)
    cleanup_after_processing: false  # Keep or delete WAV files after processing
```

**Quick Start**:
```bash
# Test API connection and process available recordings
python scripts/test_rubix44_provider.py \
    --config config/continuous_learning_config.yaml

# Run continuous learning with rubix44 provider
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db data/continuous/features.db \
    --model-dir models/continuous \
    --report-dir reports/continuous
```

**Architecture**:
```
rubix44-recorder API (10.0.0.58:5000)
       ↓ (poll every 5 min)
Rubix44DataProvider
       ↓ (download stereo WAV)
Local Download Directory
       ↓ (process channels)
StereoChannelProcessor
       ↓ (extract features)
FeatureDatabase
       ↓
Continuous Learning Pipeline
```

**Features**:
- Automatic discovery of new recordings
- State tracking to avoid reprocessing
- Health monitoring and error recovery
- Optional file cleanup after processing
- Prefix-based filtering for selective processing

#### 2. Local Directory Provider (Development/Testing)

Monitors a local directory for new WAV files (original behavior).

**Configuration**:
```yaml
orchestration:
  data_provider: local  # Use local directory watching

  local:
    audio_source_dir: /path/to/wav/files  # Or use --data-dir flag
    min_duration_sec: 60.0  # Minimum file duration
    max_files_per_day: 100  # Safety limit
    data_check_interval_minutes: 60  # Check every hour
```

**Quick Start**:
```bash
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db data/continuous/features.db \
    --model-dir models/continuous \
    --report-dir reports/continuous \
    --data-dir /path/to/wav/files
```

### Module Structure

```text
src/continuous/
├── stereo_channel_processor.py    # WAV → Features per channel
├── rubix44_data_provider.py       # Rubix44 API integration
├── feature_database.py             # SQLite/MariaDB storage
├── continuous_ingestion.py         # Local directory ingestion
├── incremental_trainer.py          # Sliding window training
├── monitoring_reporter.py          # Health monitoring & reports
└── orchestrator.py                 # 24/7 autonomous coordinator
```

### Testing

```bash
# Activate environment
source /Users/bernd/miniconda3/bin/activate xorProject

# Run all continuous learning tests
python -m pytest tests/test_continuous_learning.py -v

# Test rubix44 provider
python -m pytest tests/test_rubix44_provider.py -v

# Test rubix44 API connection only
python scripts/test_rubix44_provider.py \
    --config config/continuous_learning_config.yaml \
    --api-only
```

### Status

- ✅ **All core tests passing** (30+ tests)
- ✅ Rubix44 API integration complete
- ✅ Dual data provider support (rubix44 + local)
- ✅ State tracking and error recovery
- ⚠️ Email configuration required for alerts
- ⚠️ Visualization generation not yet implemented

For detailed status, gaps, and next steps, see **[CONTINUOUS_LEARNING_STATUS.md](docs/CONTINUOUS_LEARNING_STATUS.md)**.

## Coding Conventions and Best Practices

### File and Module Naming

#### Critical Rule: Avoid Vague Generic Names

Avoid names like "advanced", "utils2", "helpers", "misc", "common"

- ❌ BAD: `advanced_visualizations.py`, `data_utils.py`, `misc_helpers.py`
- ✅ GOOD: `embedding_analysis_plots.py`, `audio_feature_extraction.py`, `checkpoint_manager.py`

#### Rationale

Generic names don't convey what the module does. Someone reading `advanced_visualizations.py` has no idea it contains embedding comparison plots, ROC curves, and layer progression analysis. `embedding_analysis_plots.py` immediately tells you what's inside.

#### Naming Checklist

When naming modules, ask yourself: "If someone saw just the filename, would they know what functionality it provides?"

### Module Responsibilities

- **`embedding_analysis_plots.py`**: Before/after network embedding comparisons, layer-by-layer progression, ROC/PR curves, audio feature analysis, model comparison dashboards, weight evolution
- **`universal_plots.py`**: UMAP/PCA/t-SNE dimensionality reduction plots, statistical analysis (bootstrap CI, class balance)
- **`genetic_algorithm.py`**: GA optimization, hyperparameter impact visualization, progress tracking
- **`model.py`**: Neural network architecture building
- **`utils.py`**: Configuration dataclasses, path management, serialization helpers

### Code Organization Principles

1. **One module, one clear purpose**: Don't create catch-all modules
2. **Descriptive names over brevity**: `create_embedding_comparison_plots()` > `plot_embeddings()`
3. **Avoid abbreviations unless standard**: `mfcc`, `pca`, `roc` are OK; `comp`, `viz`, `proc` are not
4. **Group related functionality**: All embedding-related plots in one module
5. **Test coverage**: Every visualization function has a corresponding test

## Database Backends

The project supports both **SQLite** (local/development) and **MariaDB** (production) for storing experimental results and continuous learning features.

### Database Selection

- **SQLite**: Default for local development, single-user scenarios, and testing
  - File-based (no server required)
  - Perfect for dev/test workflows
  - Used for continuous learning when running locally

- **MariaDB**: Production backend for multi-user access and large-scale deployments
  - Networked database at 10.0.0.103
  - Connection pooling for performance
  - Better concurrency and scalability
  - Recommended for continuous learning in production

### Configuration

Database credentials are stored in `../.env`:

```bash
MARIADB_HOST=10.0.0.103
MARIADBUSER=devuser
MARIADBDEVPWD=<password>
MARIADB_DATABASE=xor_project
```

### Using MariaDB

**Initialize the database** (one-time setup):

```bash
python scripts/init_mariadb.py
```

This creates the `xor_project` database and all required tables.

**Use in code**:

```python
from continuous.feature_database import FeatureDatabase

# SQLite (default)
db = FeatureDatabase(db_path='features.db', backend='sqlite')

# MariaDB (production)
db = FeatureDatabase(backend='mariadb')
```

**Database schema**: See [docs/MARIADB_SCHEMA.md](docs/MARIADB_SCHEMA.md) for complete schema documentation.

**Connection utility**: [src/db_connection.py](src/db_connection.py) provides unified interface:

```python
from db_connection import DatabaseConnection

# Get connection with context manager
db = DatabaseConnection(backend='mariadb')
with db.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM features LIMIT 10")
    results = cursor.fetchall()
```

### Database Tables

**Continuous Learning Tables:**
- `features`: Transformed audio features (compressed numpy arrays)
- `training_runs`: Model training metadata and performance
- `validation_metrics`: Detailed validation results
- `drift_metrics`: Feature distribution drift tracking

**Experimental Results Tables:**
- `experiments`: Experiment metadata and status
- `ga_runs`: Genetic algorithm execution data
- `ga_individuals`: Individual solutions evaluated
- `model_results`: Final model performance
- `resource_usage`: CPU/memory/GPU tracking

### Migration from SQLite

To migrate existing SQLite data to MariaDB, use the migration script (when needed):

```bash
python scripts/migrate_sqlite_to_mariadb.py --sqlite-path data/features.db
```

### Backend Compatibility

The `FeatureDatabase` class automatically handles SQL dialect differences:
- Uses `?` placeholders for SQLite
- Uses `%s` placeholders for MariaDB
- Handles `AUTOINCREMENT` vs `AUTO_INCREMENT`
- Connection pooling enabled for MariaDB only
