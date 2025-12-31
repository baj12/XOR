# XOR Mode Quick Start Guide

Complete guide for using the XOR classification mode with visualization capabilities.

## What is XOR Mode?

XOR mode uses **synthetically generated 2D data** to test neural network architectures and genetic algorithm optimization. It's designed for:
- Testing network architectures before deploying on real data
- Validating GA optimization parameters
- Educational demonstrations of neural network learning
- Quick experiments with different configurations

## Quick Start

### 1. Generate XOR Data

```bash
python src/data_generator.py --config config/yaml/config_example.yaml
```

This creates:
- `data/raw/config_XXXX_data.csv` - Training data
- CSV with columns: `x, y, [noise_dims...], label`

### 2. Run Training with Visualization

```bash
python src/main.py --config config/yaml/config_example.yaml --log INFO
```

Output directory: `experiments_config_XXXX_TIMESTAMP/`

## Configuration Structure

```yaml
experiment:
  id: "518"
  description: "XOR with noise dimensions"
  noise_dimensions: 2          # Extra random features (0-10)
  class_separation: 0.75       # 0.5, 0.75, or 1.0
  use_gpu: true
  generate_advanced_viz: true  # Enable all visualizations
  viz_include_pca: true        # PCA alongside UMAP
  viz_include_tsne: false      # t-SNE (slower)
  viz_layer_progression: true  # Layer-by-layer plots
  viz_roc_curves: true         # ROC/PR curves
  viz_statistical_analysis: true
  viz_hyperparameter_impact: true

data:
  dataset_size: 10000
  class_distribution: 0.75  # Quadrant overlap
  input_dim: 4              # 2 (x,y) + noise_dimensions

ga:
  population_size: 20
  ngen: 10
  epochs: 10
  n_processes: 1  # Use 1 for GPU, 4-8 for CPU

model:
  hidden_layers: [8, 4]    # Network architecture
  activation: relu
  optimizer: adam
  lr: 0.001
  batch_size: 32
  skip_connections: null   # null, 'residual', or 'dense'
```

## Understanding XOR Data

### Quadrant Structure

```
        y
        |
   Q2   |   Q1
  (0)   |   (1)
--------|--------  x
  (1)   |   (0)
   Q3   |   Q4
        |
```

- **Class 1** (label=1): Q1 (top-right) and Q3 (bottom-left)
- **Class 0** (label=0): Q2 (top-left) and Q4 (bottom-right)

### Class Separation Parameter

- **1.0**: Complete separation, no overlap
  - Q1: x ∈ [0.25, 1], y ∈ [0.25, 1]
  - Q2: x ∈ [-1, -0.25], y ∈ [0.25, 1]

- **0.75**: Moderate overlap (default, realistic)
  - Slightly overlapping boundaries

- **0.5**: Maximum overlap
  - Boundaries touch at axes

### Noise Dimensions

Add random features to test network robustness:
- `noise_dimensions: 0` → Pure 2D XOR
- `noise_dimensions: 2` → 4D input (x, y, noise1, noise2)
- Higher values test feature selection capability

## Output Files and Visualizations

### Directory Structure

```
experiments_config_518_20250131_120000/
├── plots/
│   ├── embedding_comparison.png         # Before/after network (NEW)
│   ├── layer_progression.png            # Layer-by-layer (NEW)
│   ├── classification_curves.png        # ROC/PR curves (NEW)
│   ├── statistical_analysis.png         # Bootstrap CI (NEW)
│   ├── classification_plots_universal.png  # UMAP
│   ├── classification_plots_universal_pca.png  # PCA (if enabled)
│   ├── hyperparameter_impact.png        # GA analysis (NEW)
│   ├── accuracy_gen_X_ind_Y.png         # Training curves
│   ├── loss_gen_X_ind_Y.png
│   ├── train_test_decision_boundary.png
│   └── ga_progress_realtime.png
├── models/
│   └── model_best.keras
├── results/
│   └── results_TIMESTAMP.pkl
└── logs/
    └── experiment.log
```

### Key Visualizations Explained

#### 1. Embedding Comparison (Before/After Network)

**File**: `embedding_comparison.png`

Shows how the network transforms the feature space:

- **Top-left (UMAP - Raw)**: Input features (x, y, noise)
  - Classes overlap significantly
  - Noise creates scatter

- **Top-right (UMAP - Learned)**: Final hidden layer activations
  - Classes well-separated
  - Linear boundary possible in learned space

- **Bottom row**: Same using PCA (linear dimensionality reduction)

**What to look for**:
- Clear separation in learned representation = network learned the pattern
- Still overlapping = need deeper network or more training

#### 2. Layer Progression

**File**: `layer_progression.png`

Shows transformation through each layer:

```
Input (2D+noise) → Layer 1 (8D) → Layer 2 (4D) → Output (1D)
  [scattered]      [clustering]   [separated]    [binary]
```

**Interpretation**:
- Early layers: Partial separation begins
- Middle layers: Clear clustering emerges
- Final layer: Linear separability achieved

#### 3. Decision Boundary

**File**: `train_test_decision_boundary.png`

2D visualization of learned XOR pattern (only works for 2D input):

- **Good**: Smooth X-shaped boundary
- **Underfitting**: Straight line boundary
- **Overfitting**: Extremely wiggly boundary

#### 4. ROC and Precision-Recall Curves

**File**: `classification_curves.png`

Three panels:
- **ROC Curve**: True Positive Rate vs False Positive Rate
  - AUC = 1.0 is perfect
  - AUC = 0.5 is random guessing

- **Precision-Recall**: Especially useful for imbalanced data
  - AP (Average Precision) close to 1.0 is excellent

- **Threshold Analysis**: Optimal decision threshold
  - Find best balance of precision vs recall

#### 5. Statistical Analysis

**File**: `statistical_analysis.png`

- **Bootstrap CI**: 95% confidence interval for accuracy
- **Class Balance**: Check for imbalanced data
- **Prediction Confidence**: Well-calibrated model has high confidence on correct predictions
- **Statistical Tests**: Chi-square, KS test for distributions

#### 6. GA Progress

**File**: `ga_progress_realtime.png`

Tracks genetic algorithm convergence:
- **Fitness over generations**: Should increase
- **Distribution shift**: Population improving
- **Diversity**: Should decrease as GA converges
- **Improvement rate**: Slows down (plateau)

## Common Issues and Solutions

### Issue: Accuracy Stuck at ~50%

**Symptoms**: Random-looking decision boundary, flat accuracy curves

**Causes**:
- Learning rate too high/low
- Not enough epochs
- Network too small

**Solutions**:
```yaml
model:
  lr: 0.001           # Try 0.0001 if unstable, 0.01 if too slow
  hidden_layers: [16, 8]  # Increase from [8, 4]
ga:
  epochs: 20          # Increase from 10
```

### Issue: Overfitting (Train=95%, Val=70%)

**Symptoms**: Large gap between train/validation accuracy

**Solutions**:
- Increase dataset size: `dataset_size: 20000`
- Reduce network: `hidden_layers: [4, 4]`
- Add regularization (dropout, L2)
- Use skip connections: `skip_connections: 'residual'`

### Issue: GA Not Converging

**Symptoms**: Fitness oscillates, no improvement after many generations

**Solutions**:
```yaml
ga:
  population_size: 50    # Increase from 20
  mutpb: 0.1            # Decrease from 0.2
  ngen: 30              # More generations
```

### Issue: Visualizations Not Generated

**Check**:
1. Config has `generate_advanced_viz: true`
2. Check logs for errors: `tail -100 experiments_*/logs/experiment.log`
3. Memory sufficient (visualizations use ~2GB RAM)

## Advanced Usage

### Custom Network Architectures

```yaml
model:
  hidden_layers: [64, 32, 16, 8]  # Deeper network
  skip_connections: 'dense'       # DenseNet-style
  activation: 'tanh'              # Alternative activation
```

### Multiple Experiments

Run grid search over configurations:

```bash
for noise in 0 2 5; do
  for sep in 0.5 0.75 1.0; do
    # Generate config with these parameters
    python src/main.py --config config_noise${noise}_sep${sep}.yaml
  done
done
```

### Feature Importance Analysis

Check which features matter:
- Look at `feature_importance.png`
- For pure XOR: x and y should dominate
- Noise dimensions should have low importance

## Performance Tips

### GPU vs CPU

```yaml
experiment:
  use_gpu: true        # Faster for large networks
  # If GPU:
  ga:
    n_processes: 1     # GPU doesn't parallelize well

  # If CPU:
  # use_gpu: false
  # ga:
  #   n_processes: 8   # Parallel evaluation
```

### Speed Optimizations

1. **Reduce population**: `population_size: 10` (faster, less optimal)
2. **Fewer generations**: `ngen: 5` (quick test)
3. **Disable slow viz**: `viz_include_tsne: false`
4. **Smaller dataset**: `dataset_size: 5000` (for testing)

## Next Steps

1. **Validate setup**: Run example config, check all visualizations generate
2. **Experiment with noise**: Try `noise_dimensions: 0, 2, 5, 10`
3. **Test architectures**: Compare `[4,4]`, `[8,4]`, `[16,8]`, `[32,16,8]`
4. **Optimize GA**: Find best `population_size` and `ngen` balance
5. **Move to audio**: Once XOR works well, apply to real data

## Reference: All Config Parameters

See [CLAUDE.md](../CLAUDE.md) for complete configuration reference and [config_example.yaml](../config/yaml/config_example.yaml) for working examples.

## Troubleshooting

Check logs for detailed error messages:
```bash
tail -f experiments_config_*/logs/experiment.log
```

Common log messages:
- `"UMAP reduced XD data to 2D"` - Visualization succeeded
- `"Model not learning"` - Check learning rate
- `"Out of memory"` - Reduce `dataset_size` or `batch_size`
- `"GPU not found"` - Set `use_gpu: false`

For more help, see [CLAUDE.md](../CLAUDE.md) coding conventions and architecture details.
