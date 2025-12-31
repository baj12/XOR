# Continuous Learning for Audio Classification

**A 24/7 autonomous system for learning from streaming audio data**

---

## What Is This?

This system automatically learns to classify sounds from continuous audio recordings. Think of it as a "brain" that:

1. 📊 **Listens continuously** to stereo audio files
2. 🧠 **Learns patterns** in the audio (like presence of specific sounds)
3. 📈 **Improves over time** as new data arrives
4. 📧 **Reports its progress** via weekly emails
5. ⚠️ **Alerts you** when something unusual happens

**No manual intervention required** - it runs autonomously 24/7.

---

## Who Is This For?

- **Researchers** running long-term audio experiments
- **Biologists** monitoring animal behavior from recordings
- **Anyone** who needs to automatically classify sounds in large audio datasets

**No machine learning expertise required** to use the system!

---

## How It Works (Simple Explanation)

### Input: Stereo Audio Files

The system processes stereo (2-channel) WAV files where:
- **Left channel** = Positive class (e.g., "sound present")
- **Right channel** = Negative class (e.g., "no sound" or background)

Example uses:
- Left = animal vocalization, Right = silence
- Left = machine running, Right = machine idle
- Left = music, Right = noise

### Process: Automatic Learning

Every week, the system:

1. **Checks for new data** (new audio files)
2. **Extracts features** (converts audio to numbers computers can understand)
3. **Updates the model** (teaches the AI to recognize patterns)
4. **Tests performance** (checks if it's getting better or worse)
5. **Sends a report** (emails you a summary with charts)

### Output: Classifications + Reports

- **Real-time classification** of incoming audio
- **Weekly email reports** with performance metrics and visualizations
- **Automatic alerts** if something goes wrong (drift, performance drops)

---

## Quick Start (For Beginners)

### Step 1: Install

```bash
# Clone the repository
cd /path/to/XOR

# Activate the conda environment
conda activate xorProject

# Verify installation
python -m pytest tests/test_continuous_learning.py -v
# Should see: "30 passed"
```

### Step 2: Prepare Your Audio Files

Place your stereo WAV files in a directory:

```
/path/to/audio/
├── file1.wav  # Left=positive, Right=negative
├── file2.wav
└── file3.wav
```

**Important**: Files must be stereo (2 channels). Each channel represents a different class.

### Step 3: Configure Email (Optional but Recommended)

See [`docs/EMAIL_SETUP.md`](docs/EMAIL_SETUP.md) for detailed instructions.

Quick version:

```bash
# 1. Get Gmail app password (see email setup doc)
# 2. Set environment variable
export EMAIL_PASSWORD="your-app-password"
```

### Step 4: Run the System

```bash
python -m src.continuous.orchestrator \
    --config config/continuous_learning_config.yaml \
    --db data/continuous/features.db \
    --model-dir models/continuous \
    --report-dir reports/continuous \
    --data-dir /path/to/audio/ \
    --log INFO
```

That's it! The system is now running.

---

## What Happens Next?

### Every Hour

- System checks `/path/to/audio/` for new WAV files
- Processes any new files (extracts features, stores in database)
- Logs activity

### Every Week (Monday 2 AM)

- Trains the AI model on all data from the last 4 weeks
- Tests the model's accuracy
- Detects if data patterns are changing ("drift")

### Every Week (Monday 9 AM)

- Sends you an email report with:
  - **Accuracy**: How well the model is performing (e.g., 92.3%)
  - **Charts**: Performance trends over time
  - **Drift alerts**: If data patterns have changed
  - **Data stats**: How much audio was processed

---

## Understanding the Email Reports

Your weekly email will look like this:

```
📊 Performance Summary
- Validation Accuracy: 92.3% (↑ +1.2% from last week)
- ROC AUC: 0.965 (excellent discrimination)
- Class 0 Precision: 91.5%
- Class 1 Recall: 93.1%

📁 Data Collected This Week
- Class 0: 7 files, 7.2 minutes total
- Class 1: 7 files, 7.1 minutes total
- Total samples: 854

🧠 Model Updates
- Incremental fine-tuning completed (5 epochs, 3.2 min)
- Training accuracy: 94.1%

⚠️ Alerts
- ✓ No alerts this week

📈 Long-Term Trends
- [Chart showing accuracy over last 12 weeks]

🔍 Data Drift Analysis
- KL Divergence from baseline: 0.023 (low)
- Feature mean shift: 0.8% (normal)
- Class distribution: 51.2% / 48.8% (balanced)
```

### What to Look For

✅ **Good Signs**:
- Accuracy > 85%
- "No alerts"
- Low drift (<0.1)
- Balanced class distribution

⚠️ **Warning Signs**:
- Accuracy dropping over time
- High drift (>0.5)
- Severe class imbalance
- Storage warnings

---

## Configuration (Simple)

Edit [`config/continuous_learning_config.yaml`](config/continuous_learning_config.yaml):

### Essential Settings

```yaml
# Where new audio files arrive
orchestration:
  audio_source_dir: /path/to/your/audio/files

  # Email settings
  email_recipients:
    - your-email@example.com
  email_sender: your-email@example.com

  # Training schedule
  training_day_of_week: 0  # 0=Monday, 1=Tuesday, ..., 6=Sunday
  training_hour: 2  # 2 AM (avoid peak hours)

  # Alerts
  alert_on_drift: true  # Email if data changes
  alert_on_performance_drop: true  # Email if accuracy drops
```

### Advanced Settings (Optional)

```yaml
  # How often to check for new files
  data_check_interval_minutes: 60  # Every hour

  # Minimum data before training
  min_samples_for_training: 1000  # At least 1000 samples

  # Training strategy
  update_mode: hybrid  # incremental + periodic full retrain
  training_window_weeks: 4  # Use last 4 weeks of data
```

---

## Monitoring the System

### Check if it's running

```bash
# Look for running process
ps aux | grep orchestrator

# Check recent logs
tail -f logs/experiment.log
```

### View database stats

```python
from continuous import FeatureDatabase

db = FeatureDatabase('data/continuous/features.db')
stats = db.get_database_stats()

print(f"Total samples: {stats['total_samples']}")
print(f"Class balance: {stats['class_0_samples']} / {stats['class_1_samples']}")
print(f"Date range: {stats['earliest_timestamp']} to {stats['latest_timestamp']}")
```

### Generate visualizations manually

```python
from continuous import ContinuousVisualizer

viz = ContinuousVisualizer(
    db_path='data/continuous/features.db',
    output_dir='visualizations/continuous'
)

# Generate all plots for last 12 weeks
viz.generate_all_visualizations(weeks=12)

# Plots saved to: visualizations/continuous/
```

---

## Troubleshooting

### "No data files found"

**Problem**: System can't find audio files

**Solution**:
- Check `audio_source_dir` in config points to correct directory
- Verify WAV files are stereo (2 channels)
- Ensure WAV files are at least 1 minute long

### "Email not sent"

**Problem**: Email alerts/reports not arriving

**Solution**:
- Check `EMAIL_PASSWORD` environment variable is set
- Verify `email_recipients` in config is correct
- See [`docs/EMAIL_SETUP.md`](docs/EMAIL_SETUP.md) for detailed troubleshooting

### "Accuracy is low (<70%)"

**Problem**: Model not learning well

**Possible causes**:
- Not enough training data (need >1000 samples)
- Classes are not well-separated in audio
- Left/right channels swapped

**Solution**:
- Collect more data
- Check audio files manually - are left/right channels distinct?
- Review QC reports in `qc_reports/`

### "High drift detected"

**Problem**: Data patterns are changing over time

**This is normal if**:
- Environmental conditions change (temperature, background noise)
- Recording equipment changes
- Experimental conditions vary

**Action**: Review drift visualizations to understand what's changing

---

## Stopping the System

### Graceful shutdown

```bash
# Press Ctrl+C in the terminal running the orchestrator
# System will finish current task and shut down cleanly
```

### Force stop

```bash
# Find process ID
ps aux | grep orchestrator

# Kill process
kill <PID>
```

---

## Advanced Features

### Model Rollback

If a new model performs worse than the previous one, the system automatically reverts:

```
AUTO-ROLLBACK: Model reverted due to performance degradation
- Previous accuracy: 92.3%
- New accuracy: 87.1% (drop: 5.2%)
- Threshold: 5.0%
```

### Visualizations

The system generates:
- **Temporal UMAP**: Shows if data distribution changes over time
- **ROC Curves**: Shows discrimination ability across weeks
- **Drift Heatmaps**: Shows which features are changing
- **Performance Dashboard**: Comprehensive overview

### Model Versions

System maintains:
- `current_model.keras` - Currently deployed model
- `previous_model.keras` - Last known good model
- `safe_fallback.keras` - Baseline from full retrain

---

## Documentation

For detailed technical information:

- **[Implementation Status](docs/CONTINUOUS_LEARNING_STATUS.md)** - What's implemented, what's not
- **[Architecture Strategy](docs/CONTINUOUS_LEARNING_STRATEGY.md)** - How the system is designed
- **[Email Setup Guide](docs/EMAIL_SETUP.md)** - Detailed email configuration
- **[CLAUDE.md](CLAUDE.md)** - Development guide for AI assistants

---

## FAQ

### How much data do I need?

**Minimum**: 1 minute of audio per class (60 segments)
**Recommended**: At least 1 hour per class for good performance

### How long does training take?

**Incremental (weekly)**: 2-5 minutes
**Full retrain (monthly)**: 30-60 minutes

### Can I use mono audio files?

No, the system requires stereo files. However, you can:
1. Convert mono to stereo by duplicating the channel
2. Combine two mono files into one stereo file

### What audio formats are supported?

Currently only WAV files. Other formats (MP3, FLAC) can be converted to WAV first.

### Can I classify more than 2 classes?

Not currently. The system is designed for binary classification (2 classes only).

### How much disk space do I need?

**Features**: ~100 KB per minute of audio
**Models**: ~10 MB per model
**Database**: ~1 MB per 10,000 samples

**Example**: 100 hours of audio ≈ 600 MB (features) + 100 MB (models) + 10 MB (database) = ~710 MB

---

## Support

For questions or issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review logs: `logs/experiment.log`
3. Check [Implementation Status](docs/CONTINUOUS_LEARNING_STATUS.md)
4. Open an issue on GitHub

---

## License

[Include your license here]

---

## Citation

If you use this system in your research, please cite:

```
[Include citation information]
```

---

**Last Updated**: 2025-12-31
**Version**: 1.0.0 (Production Ready)
