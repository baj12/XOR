# Audio Classification Mode Quick Start Guide

Complete guide for audio/WAV file classification using the rubix44 dataset and other audio sources.

## What is Audio Mode?

Audio mode performs **binary classification on WAV audio files** using:
- **Feature extraction**: MFCC, spectral features, chroma, tonnetz
- **Automated QC reports**: Spectrograms, compatibility checks, waveform analysis
- **Memory-optimized processing**: Handles multi-hour, multi-GB files
- **Genetic algorithm optimization**: Finds best network architecture

Use cases:
- Sound event detection (empty cage vs stimulus)
- Music genre classification
- Speech vs non-speech
- Environmental sound classification

## Quick Start (rubix44 Dataset)

### 1. Prepare Audio Files

The `rubix44` dataset contains large WAV recordings (1-3 hours, 1-2.7 GB each):

```bash
ls data/rubix44/
# exp5.empty.noise.wav                    (Class 0: empty cage)
# lavendar.1.noise.2hrs.wav               (Class 1: lavender stimulus)
# exp6 - 1 rubix 1empty.noise.wav         (Class 0)
# lavendel in aufnahmebeaker noise...wav  (Class 1)
```

### 2. Run Classification

```bash
python src/main.py \
  --config config/yaml/audio_config.yaml \
  --audio-files \
    data/rubix44/exp5.empty.noise.wav \
    data/rubix44/lavendar.1.noise.2hrs.wav \
  --labels 0 1 \
  --log INFO
```

**Processing time**: ~15-30 minutes for 2 files (2 hours each)

### 3. Check QC Reports

QC reports generated automatically:

```bash
ls qc_reports/audio_experiment_*/
# audio_summary.csv              - File statistics
# spectrograms.png               - Frequency content
# audio_comparison.png           - Waveform comparison
# compatibility_report.png       - File compatibility check
# preprocessed_data.npz          - Extracted features
```

## Configuration Structure

```yaml
experiment:
  id: "rubix44_001"
  description: "Empty cage vs lavender stimulus"
  use_gpu: true
  generate_advanced_viz: true
  viz_include_pca: true
  viz_layer_progression: true
  viz_roc_curves: true

data:
  samples_per_file: 2000  # Random 1-sec segments per file
  input_dim: 680          # Auto-calculated from features

audio:
  sample_rate: 22050      # Downsample from 44100 for speed
  segment_duration: 1.0   # Seconds per training sample
  n_mfcc: 13              # Mel-frequency coefficients
  feature_types: ['mfcc', 'spectral']  # Feature set

  # Multi-file per class
  audio_files:
    class_0:
      paths:
        - ~/data/rubix44/exp5.empty.noise.wav
        - ~/data/rubix44/exp6 - 1 rubix 1empty.noise.wav
    class_1:
      paths:
        - ~/data/rubix44/lavendar.1.noise.2hrs.wav
        - ~/data/rubix44/lavendel in aufnahmebeaker noise...wav

ga:
  population_size: 20
  ngen: 20
  epochs: 20
  n_processes: 1  # GPU mode

model:
  hidden_layers: [128, 64, 32]  # Larger for audio
  activation: relu
  optimizer: adam
  lr: 0.001
  batch_size: 32
  skip_connections: dense  # Helps with high-dim features
```

## Audio Feature Extraction

### Feature Pipeline

```
WAV File (2.7 GB, 3 hours @ 44.1kHz stereo)
  ↓
Load in 60-sec chunks (memory efficient)
  ↓
Random sample 2000 × 1-second segments
  ↓
For each segment:
  • Convert stereo → mono (average channels)
  • Resample 44.1kHz → 22.05kHz
  • Extract MFCC (13 coefficients × 40 frames = 520 features)
  • Extract spectral features (centroid, rolloff, bandwidth, ZCR)
  ↓
Feature matrix: (2000 samples, 680 features)
  ↓
StandardScaler: Normalize to mean=0, std=1
  ↓
Save as NPZ: qc_reports/.../preprocessed_data.npz
```

### Feature Types

#### MFCC (Mel-Frequency Cepstral Coefficients)

**What**: Compact representation of spectral envelope
**Size**: 13 coefficients × 40 time frames = 520 features
**Use**: Captures perceptually-relevant frequency information

```
Audio → FFT → Mel Scale → Log → DCT → 13 coefficients
```

#### Spectral Features

**Spectral Centroid**: "Brightness" of sound
- High value → bright, high-frequency
- Low value → dark, bass-heavy

**Spectral Rolloff**: Frequency below which 85% of energy
- Indicates frequency distribution

**Spectral Bandwidth**: Spread of frequencies
- Wide → noise-like
- Narrow → tonal

**Zero Crossing Rate**: Sign changes per second
- High → noisy, unvoiced
- Low → tonal, voiced

**Total**: 4 features × 40 frames = 160 features

#### Optional Features

**Chroma**: Pitch class representation (12-dim: C, C#, ..., B)
- Useful for music classification

**Tonnetz**: Tonal centroid features (6-dim)
- Harmonic relationships

Enable in config:
```yaml
audio:
  feature_types: ['mfcc', 'spectral', 'chroma', 'tonnetz']
```

## QC Reports - What to Check

### 1. Audio Summary Table

**File**: `audio_summary.csv`, `audio_summary_table.png`

Check for:
- ✓ Sample rates match (all 44100 or all 22050)
- ✓ Channels consistent
- ✓ Similar durations (within 2× ratio)
- ✓ Similar RMS levels (within 10× ratio)
- ⚠ Dynamic range >6 dB (avoid silent files)
- ⚠ Silence % <50% (avoid dead air)

**Example**:
```
File                          | Duration | Sample Rate | RMS    | Silence%
exp5.empty.noise.wav         | 7200s    | 44100 Hz   | 0.023  | 15%
lavendar.1.noise.2hrs.wav    | 7196s    | 44100 Hz   | 0.024  | 12%
```

### 2. Spectrograms

**File**: `spectrograms.png`

Visual frequency content over time.

**Class 0 (Empty cage)**:
- Broad frequency spectrum (500-8000 Hz)
- No clear tonal structure
- Consistent noise floor

**Class 1 (Lavender stimulus)**:
- Strong harmonics (1000, 2000, 3000 Hz)
- Amplitude modulation visible
- Clearer structure than Class 0

**What to look for**:
- Are there visible differences between classes?
- If identical → classification will be difficult

### 3. Compatibility Report

**File**: `compatibility_report.png`

**Compatibility Score**:
- **100**: Perfect match ✓
- **75-99**: Good match
- **50-74**: Fair (review needed)
- **<50**: Poor (resample/normalize required)

**Checks**:
- Sample rate match
- Channel count match
- Duration ratio (~1.0)
- RMS ratio (~1.0)
- Dynamic range difference (<1 dB)

**Example** (score=100):
```json
{
  "sample_rate_match": true,
  "channels_match": true,
  "duration_ratio": 0.9947,
  "rms_ratio": 0.9992,
  "compatibility_score": 100
}
```

### 4. Data Validation

**File**: `data_validation.png`

Four panels:
1. **Class Distribution**: Should be ~50/50
2. **Feature Means**: Normal distribution centered at 0
3. **Feature Heatmap**: Varied colors = good diversity
4. **Feature Correlation**: Some correlation OK for audio

## Output Visualizations

Same as XOR mode, plus audio-specific:

### Audio Feature Analysis

**File**: `audio_feature_analysis.png`

**MFCC Heatmap**:
- X-axis: Time frames
- Y-axis: MFCC coefficient (0-12)
- Color: Coefficient value

**Interpretation**:
- Horizontal bands: Consistent frequency patterns
- Vertical lines: Transient events
- Coefficient 0 (top): Energy
- Coefficients 1-3: Main spectral shape

**Spectral Distributions**:
- Box plots per class
- Separated boxes = classes have different spectral properties ✓
- Overlapping boxes = challenging classification

### Embedding Comparison (Before/After)

**File**: `embedding_comparison.png`

**Raw Features (680D)**:
- High-dimensional MFCC + spectral
- Classes may overlap

**Learned Representation (32D)**:
- Final hidden layer activations
- Should show clear class separation

**Success indicator**: Separated clusters in learned space

## Common Issues and Solutions

### Issue: Compatibility Score <75

**Symptoms**: Red bars in compatibility report

**Causes**:
- Different sample rates
- Different equipment
- Clipped/distorted audio

**Solutions**:
```bash
# Resample to 22050 Hz
sox input.wav -r 22050 output.wav

# Normalize volume
sox input.wav output.wav norm -3

# Check for clipping
sox input.wav -n stats
```

### Issue: Classes Not Separating (Accuracy ~50%)

**Symptoms**:
- Overlapping UMAP clusters
- ROC curve near diagonal (AUC~0.5)
- Identical spectrograms

**Causes**:
- Classes acoustically too similar
- Not enough discriminative features
- Insufficient training data

**Solutions**:

1. **Check spectrograms first**: Are there visible differences?

2. **Add more features**:
```yaml
audio:
  feature_types: ['mfcc', 'spectral', 'chroma', 'tonnetz']
```

3. **Increase samples**:
```yaml
data:
  samples_per_file: 5000  # More training data
```

4. **Try longer segments**:
```yaml
audio:
  segment_duration: 2.0  # More context per sample
```

5. **Deeper network**:
```yaml
model:
  hidden_layers: [256, 128, 64, 32]
```

### Issue: Out of Memory

**Symptoms**: Process killed, MemoryError

**Causes**: Large audio files (>1 GB)

**Solutions**:
```yaml
data:
  samples_per_file: 1000  # Reduce from 2000

audio:
  segment_duration: 0.5   # Shorter segments
  sample_rate: 16000      # Lower sample rate
```

### Issue: High Silence Percentage

**Symptoms**:
- Silence % >50% in summary
- Flat waveform regions
- Low RMS level

**Solutions**:
```bash
# Trim silence from edges
sox input.wav output.wav silence 1 0.1 1% reverse silence 1 0.1 1% reverse

# Check if silence is meaningful (e.g., empty cage should have some silence)
```

## Performance and Timing

### Expected Processing Times

**Single 3-hour file** (2.7 GB, 44.1kHz stereo):
- Feature extraction: 15-20 minutes
- Training (20 generations): 30-60 minutes
- **Total**: ~45-80 minutes

**Two files** (Class 0 + Class 1):
- Feature extraction: 30-40 minutes
- Training: 30-60 minutes
- **Total**: ~60-100 minutes

### Speed Optimizations

1. **Lower sample rate**:
```yaml
audio:
  sample_rate: 16000  # From 22050 (faster, less detail)
```

2. **Fewer samples**:
```yaml
data:
  samples_per_file: 1000  # From 2000
```

3. **Essential features only**:
```yaml
audio:
  feature_types: ['mfcc', 'spectral']  # Skip chroma, tonnetz
```

4. **Quick GA test**:
```yaml
ga:
  population_size: 10
  ngen: 10
  epochs: 10
```

## Working with rubix44

### Understanding rubix44 Files

**Experimental setup**:
- Multi-hour continuous recordings
- Background noise + intermittent stimuli
- Multiple replicates per condition

**Files**:
- `exp5.empty.noise.wav`: Empty cage baseline (Class 0)
- `lavendar.1.noise.2hrs.wav`: Lavender stimulus (Class 1)
- Multiple recordings per class for robustness

### Recommended rubix44 Config

```yaml
experiment:
  id: "rubix44_full"
  use_gpu: true

data:
  samples_per_file: 2000

audio:
  sample_rate: 22050
  segment_duration: 1.0
  n_mfcc: 13
  feature_types: ['mfcc', 'spectral']

  audio_files:
    class_0:
      paths:
        - data/rubix44/exp5.empty.noise.wav
        - data/rubix44/exp6 - 1 rubix 1empty.noise.wav
    class_1:
      paths:
        - data/rubix44/lavendar.1.noise.2hrs.wav
        - data/rubix44/lavendel in aufnahmebeaker noise part - 1 rubix 1.wav

ga:
  population_size: 20
  ngen: 20
  epochs: 20
  n_processes: 1

model:
  hidden_layers: [128, 64, 32]
  skip_connections: dense
  lr: 0.001
```

### Expected Results

**Classification Performance**:
- **Accuracy**: 80-95% (depending on stimulus strength)
- **AUC**: 0.85-0.98
- **UMAP**: Clear cluster separation

**What successful classification means**:
- Network detected acoustic differences between conditions
- Lavender stimulus creates measurable frequency patterns
- Background noise distinguishable from stimulus

**If poor performance** (<70% accuracy):
- Check spectrograms for visible differences
- Stimulus presentation might be rare (low signal-to-noise)
- Background noise may mask stimulus

## Advanced Usage

### Multi-Class Classification

Extend beyond binary:

```yaml
audio_files:
  class_0: [empty_cage.wav]
  class_1: [lavender.wav]
  class_2: [peppermint.wav]
  class_3: [control.wav]
```

Update model output:
```yaml
model:
  # Will need multiclass output modification
```

### Real-Time Classification

Load trained model and classify new audio:

```python
import librosa
from tensorflow import keras
import numpy as np

# Load model
model = keras.models.load_model('experiments/.../models/model_best.keras')

# Load and process new audio
audio, sr = librosa.load('new_recording.wav', sr=22050, duration=1.0)

# Extract same features as training
# (Use audio_data_generator.py functions)
features = extract_features(audio)  # 680-dim vector

# Predict
prediction = model.predict(features.reshape(1, -1))

if prediction > 0.5:
    print("Class 1: Stimulus detected")
else:
    print("Class 0: Background noise")
```

### Ensemble Models

Combine multiple GA-trained models:

```python
models = [
    keras.models.load_model(f'run_{i}/model_best.keras')
    for i in range(5)
]

# Average predictions
predictions = np.mean([m.predict(X_test) for m in models], axis=0)
```

## File Format Requirements

### Supported Formats

- **WAV**: All sample rates, mono/stereo
- **FLAC**: Lossless compression
- **MP3**: Via librosa (converted to WAV internally)

### Automatic Preprocessing

- **Resampling**: To `audio.sample_rate`
- **Channel mixing**: Stereo → mono (average)
- **Normalization**: StandardScaler on features

### Minimum Requirements

- **Duration**: >10 seconds (for adequate sampling)
- **Quality**: Non-clipped, >6 dB dynamic range
- **Format**: Readable by librosa

## Troubleshooting

### Check Logs

```bash
tail -100 qc_reports/audio_experiment_*/validation_report.json
tail -100 experiments_*/logs/experiment.log
```

### Common Log Messages

- `"Loaded X samples from Y-hour file"` - Feature extraction succeeded
- `"Compatibility score: 100"` - Files compatible ✓
- `"High silence percentage"` - Review file quality
- `"Feature extraction failed"` - Check file format/corruption
- `"Classes not separable (AUC=0.5)"` - Check spectrograms

### GPU Issues

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If no GPU found:
# Set use_gpu: false in config
```

## Next Steps

1. **Run rubix44 example**: Verify pipeline works end-to-end
2. **Check QC reports**: Ensure files compatible and features extracted
3. **Review visualizations**: Confirm class separation in embeddings
4. **Optimize parameters**: Tune `samples_per_file`, network architecture
5. **Extend to new data**: Apply to your own audio classification tasks

## Reference

- **Full config**: See [config/yaml/audio_config.yaml](../config/yaml/audio_config.yaml)
- **Architecture**: See [CLAUDE.md](../CLAUDE.md)
- **XOR mode**: See [XOR_MODE_QUICK_START.md](XOR_MODE_QUICK_START.md) for comparison

For detailed troubleshooting and advanced features, see the comprehensive logs and [CLAUDE.md](../CLAUDE.md) documentation.
