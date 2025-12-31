# Continuous Learning Strategy for Audio Classification

## Overview

This document outlines a strategy for continuous training and validation of neural networks on streaming audio data from rubix44 and similar sources. The system should run autonomously for months, adapting to new data while tracking performance over time.

## Key Requirements

1. **Data source**: rubix44 (and future sources) delivering new WAV files continuously
2. **Autonomy**: Runs for months without intervention
3. **Monitoring**: Weekly email reports
4. **Tracking**: Longitudinal performance metrics and visualizations
5. **Storage**: Efficient data management for both raw and transformed data
6. **Minimum data**: At least 1 minute of audio per update cycle

---

## Architecture

### High-Level Flow

```
New Audio Data → Preprocessing → Feature Extraction → Model Update → Validation → Logging/Notification
                      ↓              ↓                    ↓             ↓
                   Storage       Storage            Checkpoints    Tracking DB
```

---

## 1. Data Storage Strategy

### 1.1 Raw Audio Storage

**Location**: `data/continuous/raw/YYYY-MM-DD/`

**Organization**:
```
data/continuous/raw/
├── 2025-12-31/
│   ├── class_0/
│   │   ├── 2025-12-31_00-00-00.wav  (timestamped)
│   │   ├── 2025-12-31_01-00-00.wav
│   │   └── metadata.json
│   └── class_1/
│       ├── 2025-12-31_00-15-00.wav
│       └── metadata.json
├── 2026-01-01/
│   └── ...
└── index.db  (SQLite tracking database)
```

**Retention Policy**:
- **Keep raw files**: 3 months on fast storage
- **Archive to cold storage**: After 3 months (S3, tape, etc.)
- **Reasoning**: Raw data is large (1 min @ 44.1kHz stereo = ~10 MB), but needed for:
  - Reprocessing with improved feature extraction
  - Debugging unexpected model behavior
  - Auditing/validation

**Metadata per file** (`metadata.json`):
```json
{
  "filename": "2025-12-31_00-00-00.wav",
  "timestamp": "2025-12-31T00:00:00Z",
  "duration_sec": 60.0,
  "sample_rate": 44100,
  "channels": 2,
  "file_size_bytes": 10584000,
  "md5_hash": "a1b2c3d4...",
  "source": "rubix44",
  "class_label": 0,
  "recording_conditions": {
    "temperature": 22.5,
    "experiment_id": "exp_2025_001"
  }
}
```

### 1.2 Preprocessed Feature Storage

**Location**: `data/continuous/features/YYYY-MM/`

**Organization**:
```
data/continuous/features/
├── 2025-12/
│   ├── features_2025-12-31.npz
│   │   # Contains: X (features), y (labels), timestamps, file_refs
│   ├── scaler_2025-12-31.pkl  (StandardScaler state)
│   └── feature_metadata_2025-12-31.json
├── 2026-01/
│   └── ...
└── feature_index.db
```

**Feature file format** (NPZ):
```python
# Save with:
np.savez_compressed(
    'features_2025-12-31.npz',
    X=feature_matrix,           # (n_samples, 680) - MFCC + spectral
    y=labels,                   # (n_samples,)
    timestamps=timestamps,      # (n_samples,) - when each segment was recorded
    file_refs=file_references,  # (n_samples,) - which raw file each sample came from
    segment_offsets=offsets     # (n_samples,) - offset in original file (seconds)
)
```

**Why store preprocessed features?**
- ✅ **Fast model updates**: No need to re-extract features (15-20 min → seconds)
- ✅ **Consistent preprocessing**: Same StandardScaler across time
- ✅ **Efficient storage**: 680D features << raw audio (10 MB → ~100 KB per minute)
- ✅ **Quick validation**: Can replay old data through new models
- ⚠ **Tradeoff**: If feature extraction changes, need to reprocess from raw

**Retention Policy**:
- **Keep preprocessed features**: Indefinitely (relatively small)
- **Reasoning**: Enables fast historical analysis, temporal trend detection

### 1.3 Database Schema (SQLite)

**File**: `data/continuous/tracking.db`

**Tables**:

```sql
-- Track all raw audio files
CREATE TABLE audio_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    timestamp DATETIME NOT NULL,
    class_label INTEGER NOT NULL,
    duration_sec REAL,
    sample_rate INTEGER,
    channels INTEGER,
    file_size_bytes INTEGER,
    md5_hash TEXT,
    source TEXT,  -- 'rubix44', 'rubix44_v2', etc.
    archived BOOLEAN DEFAULT 0,  -- moved to cold storage?
    archive_location TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Track preprocessed feature batches
CREATE TABLE feature_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_date DATE NOT NULL,
    filepath TEXT NOT NULL UNIQUE,
    n_samples INTEGER,
    feature_dim INTEGER,
    scaler_path TEXT,  -- path to StandardScaler
    extraction_version TEXT,  -- e.g., 'v1.0', to track if method changes
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Link features to raw files (many-to-many)
CREATE TABLE feature_to_audio (
    feature_batch_id INTEGER,
    audio_file_id INTEGER,
    sample_indices TEXT,  -- JSON array of which samples in batch came from this file
    FOREIGN KEY (feature_batch_id) REFERENCES feature_batches(id),
    FOREIGN KEY (audio_file_id) REFERENCES audio_files(id)
);

-- Track model training runs
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp DATETIME NOT NULL,
    model_path TEXT NOT NULL,
    training_data_start DATE,
    training_data_end DATE,
    n_training_samples INTEGER,
    n_validation_samples INTEGER,
    train_accuracy REAL,
    val_accuracy REAL,
    train_loss REAL,
    val_loss REAL,
    roc_auc REAL,
    config_snapshot TEXT,  -- JSON of full config
    ga_generations INTEGER,
    best_individual TEXT,  -- JSON of GA weights
    duration_seconds REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Track weekly validation runs (on held-out test set)
CREATE TABLE validation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    val_timestamp DATETIME NOT NULL,
    model_id INTEGER,  -- which model was tested
    test_data_start DATE,
    test_data_end DATE,
    n_test_samples INTEGER,
    accuracy REAL,
    precision_class_0 REAL,
    precision_class_1 REAL,
    recall_class_0 REAL,
    recall_class_1 REAL,
    f1_class_0 REAL,
    f1_class_1 REAL,
    roc_auc REAL,
    confusion_matrix TEXT,  -- JSON: [[TN, FP], [FN, TP]]
    FOREIGN KEY (model_id) REFERENCES training_runs(id)
);

-- Track data drift metrics
CREATE TABLE drift_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_timestamp DATETIME NOT NULL,
    feature_batch_id INTEGER,
    drift_score REAL,  -- e.g., KL divergence from baseline
    feature_means TEXT,  -- JSON array of mean per feature
    feature_stds TEXT,   -- JSON array of std per feature
    class_distribution TEXT,  -- JSON: {"class_0": 0.52, "class_1": 0.48}
    FOREIGN KEY (feature_batch_id) REFERENCES feature_batches(id)
);
```

---

## 2. Data Collection Strategy

### 2.1 Minimum Data Requirements

**Per update cycle**: At least 1 minute of audio per class

**Rationale**:
- 1 min @ 1-sec segments = 60 samples per class
- With class_0 and class_1: 120 samples total
- Enough for incremental model update, not full retraining

**Collection modes**:

#### Mode 1: Scheduled Collection (Recommended)
```
Daily schedule:
- 00:00-00:01: Record class_0 (empty cage baseline)
- 00:05-00:06: Record class_1 (stimulus presentation)
- Save to data/continuous/raw/YYYY-MM-DD/
```

#### Mode 2: Event-Driven Collection
```
Triggered by external event:
- Experiment start → record class_1
- Experiment end → record class_0
- Minimum 1 min per trigger
```

#### Mode 3: Continuous Streaming (Advanced)
```
Continuously record, segment later:
- 24/7 recording to circular buffer
- Label segments based on experiment log
- Extract relevant 1-min windows
```

### 2.2 Data Ingestion Pipeline

**Script**: `src/continuous/data_ingestion.py`

```python
import os
import logging
from pathlib import Path
from datetime import datetime
import hashlib
import json
import sqlite3

class ContinuousDataIngestion:
    def __init__(self, config):
        self.raw_dir = Path(config.continuous.raw_data_dir)
        self.db_path = Path(config.continuous.tracking_db)
        self.logger = logging.getLogger(__name__)

    def ingest_new_audio(self, audio_path, class_label, source='rubix44'):
        """
        Ingest a new audio file into the system.

        Args:
            audio_path: Path to WAV file
            class_label: 0 or 1
            source: Data source identifier

        Returns:
            file_id: Database ID of ingested file
        """
        # 1. Validate file
        if not self._validate_audio_file(audio_path):
            raise ValueError(f"Invalid audio file: {audio_path}")

        # 2. Compute metadata
        metadata = self._extract_metadata(audio_path, class_label, source)

        # 3. Copy to organized storage
        dest_path = self._get_destination_path(metadata)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, dest_path)

        # 4. Save metadata JSON
        metadata_path = dest_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 5. Insert into database
        file_id = self._insert_to_db(metadata, dest_path)

        self.logger.info(f"Ingested {audio_path} as file_id={file_id}")
        return file_id

    def _validate_audio_file(self, path):
        """Check if file is valid WAV with minimum duration"""
        try:
            import librosa
            y, sr = librosa.load(path, sr=None, duration=0.1)
            duration = librosa.get_duration(path=path)
            return duration >= 60.0  # At least 1 minute
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False

    def _extract_metadata(self, path, class_label, source):
        """Extract all metadata from audio file"""
        import librosa

        y, sr = librosa.load(path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)

        # Compute MD5 hash for integrity
        md5_hash = hashlib.md5(open(path, 'rb').read()).hexdigest()

        return {
            'filename': Path(path).name,
            'timestamp': datetime.now().isoformat(),
            'class_label': class_label,
            'duration_sec': duration,
            'sample_rate': sr,
            'channels': 1 if len(y.shape) == 1 else y.shape[0],
            'file_size_bytes': os.path.getsize(path),
            'md5_hash': md5_hash,
            'source': source
        }

    def _get_destination_path(self, metadata):
        """Generate organized storage path"""
        date_str = datetime.fromisoformat(metadata['timestamp']).strftime('%Y-%m-%d')
        class_dir = f"class_{metadata['class_label']}"
        return self.raw_dir / date_str / class_dir / metadata['filename']

    def _insert_to_db(self, metadata, dest_path):
        """Insert file record into tracking database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO audio_files
            (filename, filepath, timestamp, class_label, duration_sec,
             sample_rate, channels, file_size_bytes, md5_hash, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata['filename'],
            str(dest_path),
            metadata['timestamp'],
            metadata['class_label'],
            metadata['duration_sec'],
            metadata['sample_rate'],
            metadata['channels'],
            metadata['file_size_bytes'],
            metadata['md5_hash'],
            metadata['source']
        ))

        file_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return file_id
```

---

## 3. Model Update Strategy

### 3.1 Update Frequency

**Recommended**: Weekly updates

**Why weekly?**
- ✅ Accumulate 7 days × 1 min/day = 7 minutes of new data per class
- ✅ Reduces computational overhead (vs daily)
- ✅ Enough time to detect meaningful drift
- ✅ Aligns with weekly email reports

**Alternative schedules**:
- **Daily**: If data changes rapidly, need fast adaptation
- **Monthly**: If data is very stable, fewer updates needed

### 3.2 Update Modes

#### Mode A: Incremental Learning (Recommended for Continuous)

**Approach**: Fine-tune existing model on new data

```python
# Load best model from last week
model = keras.models.load_model('models/continuous/model_2025-12-24.keras')

# Load new data from this week
X_new, y_new = load_features_for_date_range('2025-12-25', '2025-12-31')

# Combine with recent historical data (sliding window)
X_historical, y_historical = load_features_for_date_range('2025-12-01', '2025-12-24')
X_train = np.vstack([X_historical, X_new])
y_train = np.hstack([y_historical, y_new])

# Fine-tune model (fewer epochs, lower learning rate)
model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy')
history = model.fit(X_train, y_train, epochs=5, validation_split=0.2)

# Save updated model
model.save(f'models/continuous/model_{datetime.now().strftime("%Y-%m-%d")}.keras')
```

**Pros**:
- Fast (minutes vs hours)
- Preserves learned representations
- Adapts to gradual drift

**Cons**:
- Can accumulate small errors over time
- May not recover from catastrophic drift

#### Mode B: Periodic Full Retraining

**Approach**: Every 4 weeks, run full GA optimization on all recent data

```python
# Every 4th week (monthly):
if week_number % 4 == 0:
    # Load all data from last 3 months
    X_all, y_all = load_features_for_date_range('2025-10-01', '2025-12-31')

    # Run full GA optimization (20 generations, population 20)
    ga = GeneticAlgorithm(config)
    best_individual, logbook = ga.run(X_all, y_all)

    # This becomes the new baseline model
    model = build_model(config.model)
    model.set_weights(best_individual)
    model.save('models/continuous/baseline_2025-12-31.keras')
```

**Pros**:
- Prevents error accumulation
- Can discover better architectures
- Resets from scratch periodically

**Cons**:
- Computationally expensive (hours)
- May temporarily decrease performance

#### Mode C: Hybrid (Recommended)

**Strategy**:
- **Weekly**: Incremental fine-tuning (Mode A)
- **Monthly**: Full GA retraining (Mode B)
- **On-demand**: If validation accuracy drops >5%, trigger full retraining

### 3.3 Training Data Window

**Sliding window approach**:

```
Week 1: Train on [Week 1]
Week 2: Train on [Week 1-2]
Week 3: Train on [Week 1-3]
Week 4: Train on [Week 1-4]
Week 5: Train on [Week 2-5]  ← Slide window, drop Week 1
Week 6: Train on [Week 3-6]
...
```

**Window size**: 4-12 weeks (configurable)

**Reasoning**:
- Too short: Model forgets older patterns
- Too long: Model doesn't adapt to new patterns
- 4-12 weeks: Good balance for most audio tasks

---

## 4. Validation Strategy

### 4.1 Held-Out Test Set

**Create initial test set** (one-time):
```python
# From initial rubix44 data, set aside 20% as permanent test set
X_test, y_test = load_rubix44_data()
X_test, y_test = stratified_split(X_test, y_test, test_size=0.2)

# Save and NEVER train on this
np.savez('data/continuous/permanent_test_set.npz', X=X_test, y=y_test)
```

**Purpose**:
- Unbiased evaluation across all time
- Detect if model is improving or degrading
- Not contaminated by continuous learning

### 4.2 Weekly Validation Protocol

**Every week** (after model update):

```python
# 1. Load permanent test set
test_data = np.load('data/continuous/permanent_test_set.npz')
X_test, y_test = test_data['X'], test_data['y']

# 2. Load updated model
model = keras.models.load_model('models/continuous/model_2025-12-31.keras')

# 3. Predict
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# 4. Compute metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 5. Log to database
log_validation_run(model_id, accuracy, precision, recall, f1, roc_auc)

# 6. Check for alerts
if accuracy < 0.7:  # Threshold
    send_alert_email("Model accuracy dropped below 70%!")
```

### 4.3 Rolling Validation (Recent Data)

**In addition to permanent test set**, validate on recent held-out data:

```python
# Every week, test on data from 2 weeks ago (not used in training)
X_recent_test, y_recent_test = load_features_for_date_range('2025-12-17', '2025-12-23')

# This tests if model works well on "unseen but recent" data
# Helps detect if model is overfitting to older data
```

---

## 5. Projection Plot Management

### 5.1 What to Track Over Time

**Visualizations to generate weekly**:

1. **Embedding comparison** (before/after network)
   - How well does current model separate classes?
   - Compare to previous weeks

2. **UMAP/PCA projection** with temporal color coding
   - Color points by week collected
   - Detect if data distribution is drifting

3. **ROC curves**
   - Overlay curves from each week
   - Track AUC progression

4. **Confusion matrix heatmap**
   - Show errors over time

5. **Feature drift plots**
   - Mean/std of each feature dimension over time
   - Detect sensor degradation or environmental changes

### 5.2 Storage Organization

```
visualizations/continuous/
├── weekly/
│   ├── 2025-W52/  (ISO week number)
│   │   ├── embedding_comparison.png
│   │   ├── umap_temporal.png
│   │   ├── roc_curves_historical.png
│   │   ├── confusion_matrix.png
│   │   └── feature_drift.png
│   ├── 2026-W01/
│   │   └── ...
├── monthly/
│   ├── 2025-12/
│   │   ├── monthly_summary_dashboard.png  (4-week overview)
│   │   └── performance_trends.png
└── animations/
    └── umap_evolution_2025.mp4  (time-lapse of UMAP over year)
```

### 5.3 Temporal UMAP Visualization

**New visualization**: UMAP colored by data collection time

```python
def create_temporal_umap_plot(all_data_by_week, model, save_path):
    """
    Create UMAP where points are colored by collection week.
    Shows if data distribution drifts over time.
    """
    import umap
    import matplotlib.pyplot as plt
    from matplotlib import cm

    # Combine all weeks
    X_all = []
    week_labels = []

    for week_num, (X_week, y_week) in enumerate(all_data_by_week):
        X_all.append(X_week)
        week_labels.extend([week_num] * len(X_week))

    X_all = np.vstack(X_all)
    week_labels = np.array(week_labels)

    # UMAP projection
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_all)

    # Plot with week-based colormap
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=week_labels,
        cmap='viridis',
        alpha=0.6,
        s=20
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Week Number', rotation=270, labelpad=20)

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Data Distribution Over Time (UMAP Projection)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
```

**Interpretation**:
- **Clustered by week**: Data is drifting (environmental changes, sensor drift)
- **Well-mixed**: Data distribution is stable over time ✓

### 5.4 Animated Evolution

**Create time-lapse video** (monthly):

```python
# Create animation showing how UMAP projection evolves
import matplotlib.animation as animation

fig, ax = plt.subplots()

def update_frame(week_num):
    ax.clear()
    X_upto_week = data_upto_week(week_num)
    X_umap = umap.UMAP().fit_transform(X_upto_week)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='coolwarm')
    ax.set_title(f'Week {week_num}')

anim = animation.FuncAnimation(fig, update_frame, frames=52, interval=200)
anim.save('visualizations/continuous/animations/umap_evolution_2025.mp4')
```

---

## 6. Monitoring and Alerting

### 6.1 Weekly Email Report

**Script**: `src/continuous/weekly_report.py`

**Email content** (HTML):

```html
<h1>Weekly Audio Classification Report</h1>
<h2>Week 2025-W52 (Dec 25-31)</h2>

<h3>📊 Performance Summary</h3>
<table>
  <tr><td>Validation Accuracy</td><td>92.3%</td><td>⬆️ +1.2%</td></tr>
  <tr><td>ROC AUC</td><td>0.965</td><td>⬆️ +0.008</td></tr>
  <tr><td>Class 0 Precision</td><td>91.5%</td><td>➡️ 0.0%</td></tr>
  <tr><td>Class 1 Recall</td><td>93.1%</td><td>⬆️ +2.1%</td></tr>
</table>

<h3>📁 Data Collected This Week</h3>
<ul>
  <li>Class 0: 7 files, 7.2 minutes total</li>
  <li>Class 1: 7 files, 7.1 minutes total</li>
  <li>Total samples: 854 (after segmentation)</li>
</ul>

<h3>🧠 Model Updates</h3>
<ul>
  <li>Incremental fine-tuning completed (5 epochs, 3.2 min)</li>
  <li>Training accuracy: 94.1%</li>
  <li>Model saved: models/continuous/model_2025-12-31.keras</li>
</ul>

<h3>📈 Visualizations</h3>
<img src="cid:embedding_comparison.png" width="600">
<img src="cid:roc_curves_historical.png" width="600">

<h3>⚠️ Alerts</h3>
<p style="color: green;">✓ No alerts this week</p>

<h3>📊 Long-Term Trends (Last 12 Weeks)</h3>
<img src="cid:performance_trends.png" width="600">

<h3>🔍 Data Drift Analysis</h3>
<ul>
  <li>KL Divergence from baseline: 0.023 (low)</li>
  <li>Feature mean shift: 0.8% (normal)</li>
  <li>Class distribution: 51.2% / 48.8% (balanced)</li>
</ul>

<hr>
<p><small>Generated automatically by continuous_learning_system.py</small></p>
```

**Python implementation**:

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

def send_weekly_report(week_num, metrics, visualizations):
    """
    Send HTML email with embedded images.

    Args:
        week_num: ISO week number (e.g., '2025-W52')
        metrics: Dict of performance metrics
        visualizations: Dict of {name: image_path}
    """
    msg = MIMEMultipart('related')
    msg['Subject'] = f'Audio Classification Report - {week_num}'
    msg['From'] = 'audio-classifier@example.com'
    msg['To'] = 'team@example.com'

    # HTML body
    html_body = generate_html_report(week_num, metrics)
    msg.attach(MIMEText(html_body, 'html'))

    # Embed images
    for img_name, img_path in visualizations.items():
        with open(img_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-ID', f'<{img_name}>')
            msg.attach(img)

    # Send
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()
        server.login('user', 'password')
        server.send_message(msg)
```

### 6.2 Alert Conditions

**Trigger alerts** (immediate email, not weekly) if:

```python
# 1. Accuracy drop
if current_accuracy < previous_accuracy - 0.05:  # 5% drop
    send_alert("Accuracy dropped significantly!")

# 2. Data quality issues
if silence_percentage > 0.5:  # 50% silence in new audio
    send_alert("High silence in new recordings - check microphone")

# 3. Class imbalance
if class_ratio < 0.3 or class_ratio > 0.7:
    send_alert("Severe class imbalance detected")

# 4. Feature drift
if kl_divergence(current_features, baseline_features) > 0.5:
    send_alert("Significant feature drift detected - environmental change?")

# 5. Model uncertainty
if np.mean(prediction_confidence) < 0.6:  # Low confidence
    send_alert("Model showing high uncertainty - may need retraining")

# 6. Storage issues
if disk_usage_percent > 90:
    send_alert("Disk space running low - archive old data")
```

---

## 7. Change Tracking Over Time

### 7.1 Performance Tracking Database Queries

**Get performance trend**:
```sql
SELECT
    DATE(val_timestamp) as date,
    accuracy,
    roc_auc,
    (accuracy - LAG(accuracy) OVER (ORDER BY val_timestamp)) as accuracy_delta
FROM validation_runs
ORDER BY val_timestamp DESC
LIMIT 52;  -- Last 52 weeks
```

**Detect sudden drops**:
```sql
SELECT *
FROM validation_runs
WHERE accuracy < (
    SELECT AVG(accuracy) - 2*STDDEV(accuracy)
    FROM validation_runs
    WHERE val_timestamp > DATE('now', '-12 weeks')
)
ORDER BY val_timestamp DESC;
```

### 7.2 Feature Drift Tracking

**Compute drift metrics weekly**:

```python
def compute_feature_drift(X_baseline, X_current):
    """
    Compare current features to baseline distribution.

    Returns:
        drift_metrics: Dict with various drift indicators
    """
    from scipy.stats import ks_2samp, entropy

    drift_metrics = {
        'kl_divergence': [],
        'ks_statistic': [],
        'mean_shift': [],
        'std_shift': []
    }

    n_features = X_baseline.shape[1]

    for i in range(n_features):
        # KL divergence (binned distributions)
        hist_baseline, bins = np.histogram(X_baseline[:, i], bins=50, density=True)
        hist_current, _ = np.histogram(X_current[:, i], bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        hist_baseline += 1e-10
        hist_current += 1e-10

        kl_div = entropy(hist_baseline, hist_current)
        drift_metrics['kl_divergence'].append(kl_div)

        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(X_baseline[:, i], X_current[:, i])
        drift_metrics['ks_statistic'].append(ks_stat)

        # Mean/std shifts
        mean_shift = abs(np.mean(X_current[:, i]) - np.mean(X_baseline[:, i]))
        std_shift = abs(np.std(X_current[:, i]) - np.std(X_baseline[:, i]))

        drift_metrics['mean_shift'].append(mean_shift)
        drift_metrics['std_shift'].append(std_shift)

    # Overall drift score (average KL divergence)
    drift_metrics['overall_score'] = np.mean(drift_metrics['kl_divergence'])

    return drift_metrics
```

**Visualize drift over time**:

```python
def plot_drift_trends(drift_history, save_path):
    """
    Plot how features drift over time.

    Args:
        drift_history: List of (week, drift_metrics) tuples
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    weeks = [d[0] for d in drift_history]
    overall_scores = [d[1]['overall_score'] for d in drift_history]

    # 1. Overall drift score over time
    axes[0, 0].plot(weeks, overall_scores, marker='o')
    axes[0, 0].axhline(0.1, color='orange', linestyle='--', label='Warning threshold')
    axes[0, 0].axhline(0.5, color='red', linestyle='--', label='Critical threshold')
    axes[0, 0].set_xlabel('Week')
    axes[0, 0].set_ylabel('Overall Drift Score (KL Div)')
    axes[0, 0].set_title('Feature Drift Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Heatmap of per-feature KL divergence
    kl_matrix = np.array([d[1]['kl_divergence'] for d in drift_history])
    im = axes[0, 1].imshow(kl_matrix.T, aspect='auto', cmap='YlOrRd')
    axes[0, 1].set_xlabel('Week')
    axes[0, 1].set_ylabel('Feature Index')
    axes[0, 1].set_title('Per-Feature Drift (KL Divergence)')
    plt.colorbar(im, ax=axes[0, 1])

    # 3. Mean shift trends (top 10 drifting features)
    mean_shifts = np.array([d[1]['mean_shift'] for d in drift_history])
    top_drifting = np.argsort(mean_shifts[-1])[-10:]  # Top 10 from latest week

    for feat_idx in top_drifting:
        axes[1, 0].plot(weeks, mean_shifts[:, feat_idx], label=f'Feat {feat_idx}')

    axes[1, 0].set_xlabel('Week')
    axes[1, 0].set_ylabel('Mean Shift')
    axes[1, 0].set_title('Top 10 Drifting Features (Mean Shift)')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Distribution of drift scores
    axes[1, 1].hist(drift_history[-1][1]['kl_divergence'], bins=30, alpha=0.7)
    axes[1, 1].set_xlabel('KL Divergence')
    axes[1, 1].set_ylabel('Number of Features')
    axes[1, 1].set_title('Current Week: Distribution of Feature Drift')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
```

### 7.3 Model Evolution Tracking

**Track model checkpoints**:

```
models/continuous/
├── baselines/
│   ├── baseline_2025-10-01.keras  (monthly full retrain)
│   ├── baseline_2025-11-01.keras
│   └── baseline_2025-12-01.keras
├── weekly/
│   ├── model_2025-10-07.keras
│   ├── model_2025-10-14.keras
│   └── ...
└── metadata/
    ├── training_log.jsonl  (one JSON object per line, append-only)
    └── model_comparison.csv
```

**Training log format** (JSONL):

```jsonl
{"timestamp": "2025-10-07T12:00:00Z", "model": "model_2025-10-07.keras", "mode": "incremental", "train_acc": 0.921, "val_acc": 0.903, "duration_sec": 180}
{"timestamp": "2025-10-14T12:00:00Z", "model": "model_2025-10-14.keras", "mode": "incremental", "train_acc": 0.925, "val_acc": 0.907, "duration_sec": 175}
{"timestamp": "2025-11-01T12:00:00Z", "model": "baseline_2025-11-01.keras", "mode": "full_retrain", "train_acc": 0.932, "val_acc": 0.918, "ga_generations": 20, "duration_sec": 7200}
```

**Visualize model evolution**:

```python
def plot_model_evolution(training_log_path, save_path):
    """Plot how models improve over time"""
    import pandas as pd

    # Load training log
    df = pd.read_json(training_log_path, lines=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Accuracy over time
    axes[0].plot(df['timestamp'], df['train_acc'], label='Train Accuracy', marker='o')
    axes[0].plot(df['timestamp'], df['val_acc'], label='Val Accuracy', marker='s')

    # Highlight full retrains
    full_retrains = df[df['mode'] == 'full_retrain']
    axes[0].scatter(full_retrains['timestamp'], full_retrains['val_acc'],
                   color='red', s=100, marker='*', label='Full Retrain', zorder=5)

    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Performance Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Training duration over time
    axes[1].bar(df['timestamp'], df['duration_sec'] / 60,
               color=['red' if mode == 'full_retrain' else 'blue' for mode in df['mode']],
               alpha=0.7)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Training Duration (minutes)')
    axes[1].set_title('Training Time Per Update')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
```

---

## 8. System Architecture

### 8.1 Cron Jobs (Linux/Mac) or Task Scheduler (Windows)

**Crontab entries**:

```bash
# Daily: Ingest new audio (runs at 1 AM)
0 1 * * * /path/to/venv/bin/python /path/to/src/continuous/ingest_daily_audio.py

# Weekly: Model update (runs Monday 2 AM)
0 2 * * 1 /path/to/venv/bin/python /path/to/src/continuous/weekly_model_update.py

# Weekly: Generate report and send email (runs Monday 3 AM, after update)
0 3 * * 1 /path/to/venv/bin/python /path/to/src/continuous/weekly_report.py

# Monthly: Full GA retrain (runs 1st of month, 3 AM)
0 3 1 * * /path/to/venv/bin/python /path/to/src/continuous/monthly_full_retrain.py

# Daily: Check disk space and archive old data (runs at 4 AM)
0 4 * * * /path/to/venv/bin/python /path/to/src/continuous/archive_old_data.py
```

### 8.2 Main Orchestration Script

**File**: `src/continuous/orchestrator.py`

```python
#!/usr/bin/env python
"""
Continuous Learning Orchestrator

Coordinates all continuous learning tasks:
- Data ingestion
- Feature extraction
- Model updates
- Validation
- Drift detection
- Reporting
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import all continuous learning modules
from data_ingestion import ContinuousDataIngestion
from feature_extraction import ContinuousFeatureExtraction
from model_trainer import ContinuousModelTrainer
from validator import ContinuousValidator
from drift_detector import DriftDetector
from reporter import WeeklyReporter

class ContinuousLearningOrchestrator:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.logger = self.setup_logging()

        # Initialize subsystems
        self.ingestion = ContinuousDataIngestion(self.config)
        self.feature_extractor = ContinuousFeatureExtraction(self.config)
        self.trainer = ContinuousModelTrainer(self.config)
        self.validator = ContinuousValidator(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.reporter = WeeklyReporter(self.config)

    def run_daily_ingestion(self):
        """Run daily data ingestion"""
        self.logger.info("Starting daily data ingestion...")

        # Get new audio files from source
        new_files = self.get_new_audio_files()

        for audio_file, class_label in new_files:
            try:
                file_id = self.ingestion.ingest_new_audio(audio_file, class_label)
                self.logger.info(f"Ingested {audio_file} as file_id={file_id}")
            except Exception as e:
                self.logger.error(f"Failed to ingest {audio_file}: {e}")
                self.send_alert(f"Ingestion failed: {audio_file}")

        # Extract features from new files
        self.feature_extractor.process_new_files(new_files)

        self.logger.info(f"Daily ingestion complete. Processed {len(new_files)} files.")

    def run_weekly_update(self):
        """Run weekly model update and validation"""
        self.logger.info("Starting weekly model update...")

        # 1. Load data from last 4 weeks (sliding window)
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=4)
        X_train, y_train = self.feature_extractor.load_features_for_range(start_date, end_date)

        # 2. Incremental model update
        updated_model = self.trainer.incremental_update(X_train, y_train)

        # 3. Validate on permanent test set
        metrics = self.validator.validate(updated_model)

        # 4. Detect drift
        drift_metrics = self.drift_detector.compute_drift(X_train)

        # 5. Check for alerts
        self.check_alert_conditions(metrics, drift_metrics)

        # 6. Generate visualizations
        viz_paths = self.generate_weekly_visualizations(updated_model, metrics, drift_metrics)

        # 7. Send weekly report
        self.reporter.send_weekly_report(metrics, drift_metrics, viz_paths)

        self.logger.info("Weekly update complete.")

    def run_monthly_full_retrain(self):
        """Run monthly full GA retraining"""
        self.logger.info("Starting monthly full retraining...")

        # Load all data from last 3 months
        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=12)
        X_all, y_all = self.feature_extractor.load_features_for_range(start_date, end_date)

        # Run full GA optimization
        best_model = self.trainer.full_retrain(X_all, y_all)

        # Validate
        metrics = self.validator.validate(best_model)

        # Save as new baseline
        baseline_path = f"models/continuous/baselines/baseline_{datetime.now().strftime('%Y-%m-%d')}.keras"
        best_model.save(baseline_path)

        self.logger.info(f"Monthly retrain complete. Baseline saved to {baseline_path}")

    def get_new_audio_files(self):
        """
        Get new audio files from source.

        This is placeholder - actual implementation depends on how rubix44 delivers data:
        - Option 1: Watch directory for new files
        - Option 2: API call to rubix44 system
        - Option 3: Database query
        """
        # Placeholder
        watch_dir = Path(self.config.continuous.audio_source_dir)
        new_files = []

        for file in watch_dir.glob('*.wav'):
            # Check if already ingested
            if not self.ingestion.is_already_ingested(file):
                # Infer class label from filename or metadata
                class_label = self.infer_class_label(file)
                new_files.append((file, class_label))

        return new_files

    def check_alert_conditions(self, metrics, drift_metrics):
        """Check if any alert conditions are met"""
        alerts = []

        if metrics['accuracy'] < 0.7:
            alerts.append(f"Accuracy dropped to {metrics['accuracy']:.2%}")

        if drift_metrics['overall_score'] > 0.5:
            alerts.append(f"High feature drift detected: {drift_metrics['overall_score']:.3f}")

        if len(alerts) > 0:
            self.send_alert("\n".join(alerts))

    def send_alert(self, message):
        """Send immediate alert email"""
        self.reporter.send_alert_email(message)
        self.logger.warning(f"ALERT: {message}")

def main():
    parser = argparse.ArgumentParser(description='Continuous Learning Orchestrator')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--mode', choices=['daily', 'weekly', 'monthly'], required=True)

    args = parser.parse_args()

    orchestrator = ContinuousLearningOrchestrator(args.config)

    if args.mode == 'daily':
        orchestrator.run_daily_ingestion()
    elif args.mode == 'weekly':
        orchestrator.run_weekly_update()
    elif args.mode == 'monthly':
        orchestrator.run_monthly_full_retrain()

if __name__ == '__main__':
    main()
```

---

## 9. Configuration

**File**: `config/continuous_learning_config.yaml`

```yaml
continuous:
  # Data storage
  raw_data_dir: "data/continuous/raw"
  features_dir: "data/continuous/features"
  tracking_db: "data/continuous/tracking.db"

  # Audio source (how to get new data)
  audio_source_dir: "/path/to/rubix44/output"
  audio_source_type: "directory_watch"  # or "api", "database"

  # Data collection
  min_duration_sec: 60.0  # Minimum 1 minute per file
  max_files_per_day: 10   # Safety limit

  # Feature extraction
  feature_extraction_version: "v1.0"
  samples_per_file: 60    # 60 segments from 1 min = 1 sample/sec

  # Training schedule
  update_frequency: "weekly"  # daily, weekly, monthly
  training_window_weeks: 4    # Use last 4 weeks of data

  # Model update strategy
  update_mode: "hybrid"  # incremental, full_retrain, hybrid
  incremental_epochs: 5
  incremental_lr: 0.0001
  full_retrain_frequency_weeks: 4

  # Validation
  permanent_test_set: "data/continuous/permanent_test_set.npz"
  validation_frequency: "weekly"

  # Alerts
  accuracy_drop_threshold: 0.05  # Alert if accuracy drops >5%
  drift_score_threshold: 0.5
  alert_email: "team@example.com"

  # Reporting
  weekly_report_email: "team@example.com"
  report_day: "Monday"
  report_time: "09:00"

  # Storage management
  raw_retention_days: 90      # Keep raw files for 3 months
  archive_location: "/mnt/archive"
  disk_warning_threshold: 0.9  # Warn at 90% full

  # Visualization
  visualizations_dir: "visualizations/continuous"
  create_animations: true
  animation_frequency: "monthly"

# Audio processing (same as before)
audio:
  sample_rate: 22050
  segment_duration: 1.0
  n_mfcc: 13
  feature_types: ['mfcc', 'spectral']

# Model config (same as before)
model:
  hidden_layers: [128, 64, 32]
  activation: relu
  optimizer: adam
  lr: 0.001
  batch_size: 32
  skip_connections: dense

# GA config for monthly retraining
ga:
  population_size: 20
  ngen: 20
  epochs: 20
  n_processes: 1
```

---

## 10. What You Might Be Missing

### 10.1 Data Provenance and Reproducibility

**Track full lineage**:
```
Model X was trained on:
  - Data from 2025-10-01 to 2025-10-31
  - Feature extraction version v1.0
  - StandardScaler fitted on 2025-10-01
  - Config snapshot: {...}
  - Parent model: baseline_2025-10-01.keras
  - Update mode: incremental
```

**Why important**:
- Reproduce any model exactly
- Debug unexpected behavior
- Regulatory compliance (if medical/safety-critical)

### 10.2 Model Versioning

Use semantic versioning for models:
- `v1.0.0`: Initial baseline
- `v1.1.0`: Weekly incremental update
- `v2.0.0`: Monthly full retrain (breaking change)

### 10.3 A/B Testing

**Deploy two models simultaneously**:
```python
# Production: current_model (v1.5.0)
# Shadow: new_model (v1.6.0)

# On new data, get predictions from both
pred_current = current_model.predict(X_new)
pred_shadow = new_model.predict(X_new)

# Log both predictions
# Compare performance before switching
```

**Switch to new model** only if shadow model outperforms for 1 week.

### 10.4 Model Rollback

**Keep last N working models**:
```
models/continuous/production/
├── current_model.keras -> weekly/model_2025-12-31.keras
├── previous_model.keras -> weekly/model_2025-12-24.keras
└── safe_fallback.keras -> baselines/baseline_2025-12-01.keras
```

**If new model fails**, automatically rollback:
```python
if new_model_accuracy < old_model_accuracy - 0.05:
    logger.warning("New model underperforming, rolling back")
    shutil.copy('previous_model.keras', 'current_model.keras')
    send_alert("Model rollback triggered")
```

### 10.5 Explainability Tracking

**Track which features matter over time**:
```python
# Every week, compute feature importance
importances = compute_feature_importance(model)

# Save to database
save_feature_importance(week_num, importances)

# Alert if important features change drastically
if correlation(importances_this_week, importances_last_week) < 0.7:
    send_alert("Model is relying on different features - investigate!")
```

### 10.6 External Validation

**Collect ground truth labels**:
```python
# Every month, manually label 100 random samples
# Compare model predictions to human labels
# Compute inter-rater agreement

human_labels = get_human_labels(sample_ids)
model_preds = model.predict(samples)

agreement = accuracy_score(human_labels, model_preds)

if agreement < 0.85:
    send_alert("Model diverging from human labels")
```

### 10.7 Computational Budget

**Track resource usage**:
```python
# Log CPU/GPU time, memory, disk I/O
resource_log = {
    'timestamp': datetime.now(),
    'task': 'weekly_update',
    'cpu_seconds': 180,
    'gpu_seconds': 45,
    'peak_memory_gb': 8.2,
    'disk_read_gb': 1.5,
    'disk_write_gb': 0.3
}

# Alert if resource usage spikes
if cpu_seconds > 600:  # More than 10 min
    send_alert("Training taking unusually long")
```

### 10.8 Backup and Disaster Recovery

**Automated backups**:
```bash
# Daily: Backup database
0 5 * * * rsync -av /path/to/tracking.db /backup/tracking_$(date +\%Y-\%m-\%d).db

# Weekly: Backup models
0 6 * * 1 rsync -av models/continuous/ /backup/models/

# Monthly: Backup all features
0 7 1 * * rsync -av data/continuous/features/ /backup/features/
```

**Test restore procedure monthly**.

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create database schema
- [ ] Implement data ingestion pipeline
- [ ] Set up directory structure
- [ ] Write data validation functions

### Phase 2: Core Pipeline (Weeks 3-4)
- [ ] Implement feature extraction module
- [ ] Create incremental training function
- [ ] Build validation framework
- [ ] Set up basic logging

### Phase 3: Monitoring (Weeks 5-6)
- [ ] Implement drift detection
- [ ] Create weekly report generator
- [ ] Set up email notifications
- [ ] Build alert system

### Phase 4: Visualization (Weeks 7-8)
- [ ] Temporal UMAP plots
- [ ] Performance trend dashboards
- [ ] Feature drift visualizations
- [ ] Create animation pipeline

### Phase 5: Production (Weeks 9-10)
- [ ] Set up cron jobs
- [ ] Implement orchestrator
- [ ] Test end-to-end pipeline
- [ ] Create runbooks and documentation

### Phase 6: Optimization (Weeks 11-12)
- [ ] Add A/B testing
- [ ] Implement model rollback
- [ ] Optimize storage (compression, archival)
- [ ] Performance tuning

---

## 12. Summary

**Key Decisions**:

| Aspect | Recommendation | Rationale |
|--------|---------------|-----------|
| **Raw data storage** | 3 months local, then archive | Balance accessibility vs cost |
| **Feature storage** | Indefinite | Small, enables fast analysis |
| **Update frequency** | Weekly | Good balance speed/stability |
| **Update mode** | Hybrid (weekly incremental + monthly full) | Adapt quickly, reset periodically |
| **Training window** | 4-12 weeks sliding | Enough history, adapts to change |
| **Validation** | Permanent test set + rolling | Unbiased + recent performance |
| **Monitoring** | Weekly reports + instant alerts | Proactive issue detection |
| **Visualization** | Store all, animate monthly | Track evolution, easy interpretation |

**Critical Success Factors**:
1. **Automated monitoring** - System must self-diagnose issues
2. **Data provenance** - Every model traceable to exact data
3. **Gradual adaptation** - Balance stability vs adaptation
4. **Resource management** - Archive old data, manage disk space
5. **Human oversight** - Weekly reports keep humans in loop

**Next Steps**:
1. Review this strategy document
2. Decide on implementation timeline
3. Start with Phase 1 (database + ingestion)
4. Iterate based on real-world experience
