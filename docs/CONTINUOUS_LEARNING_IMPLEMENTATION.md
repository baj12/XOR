# Continuous Learning Implementation - Stereo WAV Streams

## Overview

This implementation handles continuous audio classification from stereo WAV files where:
- **Channel 1 (Left)**: Positive class samples
- **Channel 2 (Right)**: Negative class samples
- **Storage**: 2TB database for transformed features only (no raw audio storage)
- **Goal**: Maximize data capture (as many samples as possible per day)
- **Validation**: No human validation needed (labels determined by channel)

## Storage Capacity Analysis

### 2TB Database Storage Calculations

**Feature dimensions per sample**:
- MFCC: 13 coefficients × 40 frames = 520 features
- Spectral: 4 features × 40 frames = 160 features
- **Total**: 680 features per 1-second segment

**Storage per sample**:
- Features: 680 × 8 bytes (float64) = 5,440 bytes = 5.44 KB
- Label: 1 × 8 bytes (int64) = 8 bytes
- Timestamp: 8 bytes (datetime64)
- Metadata: ~100 bytes (file_ref, segment_offset, etc.)
- **Total per sample**: ~5.56 KB

**Database overhead**:
- SQLite index overhead: ~20%
- Total storage per sample: 5.56 KB × 1.2 = **6.67 KB**

**Capacity calculations**:

```
2 TB = 2,000 GB = 2,000,000 MB = 2,048,000,000 KB

Maximum samples = 2,048,000,000 KB ÷ 6.67 KB/sample = 307,016,049 samples

At 1 sample per second:
- Total duration: 307,016,049 seconds
- = 5,116,934 minutes
- = 85,282 hours
- = 3,553 days
- = 9.7 years of continuous audio

With stereo (2 channels):
- = 4.85 years of stereo audio (both channels processed)
```

**Daily capture capacity**:

```
Target: Maximize samples per day

Scenario 1: 1 hour/day continuous recording
- 1 hour = 3,600 seconds
- Stereo: 3,600 samples × 2 channels = 7,200 samples/day
- Storage: 7,200 × 6.67 KB = 48 MB/day
- 2 TB lasts: 2,000,000 MB ÷ 48 MB/day = 41,667 days = 114 years

Scenario 2: 8 hours/day continuous recording
- 8 hours = 28,800 seconds
- Stereo: 28,800 × 2 = 57,600 samples/day
- Storage: 57,600 × 6.67 KB = 384 MB/day
- 2 TB lasts: 2,000,000 MB ÷ 384 MB/day = 5,208 days = 14.3 years

Scenario 3: 24/7 continuous recording
- 24 hours = 86,400 seconds
- Stereo: 86,400 × 2 = 172,800 samples/day
- Storage: 172,800 × 6.67 KB = 1,153 MB/day = 1.13 GB/day
- 2 TB lasts: 2,000 GB ÷ 1.13 GB/day = 1,770 days = 4.85 years
```

**Recommendation**: **24/7 continuous recording**
- Maximizes data capture
- 2TB provides ~5 years of storage
- No raw audio storage needed (features only)
- Can process retrospectively if needed

---

## Simplified Architecture

### Data Flow

```
Stereo WAV File (live or simulated)
    ↓
Channel Separator
    ├─ Left Channel  → Feature Extraction → Label: 1 (positive) ─┐
    └─ Right Channel → Feature Extraction → Label: 0 (negative) ─┤
                                                                   ↓
                                            Feature Database (2TB)
                                                    ↓
                                    [Weekly] Load Recent Features
                                                    ↓
                                        Incremental Model Update
                                                    ↓
                                        Validation & Monitoring
                                                    ↓
                                            Weekly Email Report
```

### Key Simplifications

1. **No raw audio storage** - Only transformed features
2. **No human validation** - Labels from channel assignment
3. **Single database** - All features + metadata in one SQLite DB
4. **Maximize capture** - 24/7 recording if possible

---

## Database Schema (Optimized)

**Single SQLite file**: `continuous_learning.db` (on 2TB storage server)

```sql
-- Main feature storage table
CREATE TABLE features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    class_label INTEGER NOT NULL,  -- 0 (negative/right) or 1 (positive/left)
    channel INTEGER NOT NULL,       -- 0 (right) or 1 (left)
    source_file TEXT,               -- Original WAV filename
    segment_offset_sec REAL,        -- Position in original file

    -- Feature blob (680 floats = 5.44 KB)
    features BLOB NOT NULL,         -- Compressed numpy array

    -- Metadata
    extraction_version TEXT DEFAULT 'v1.0',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    -- Indexes for fast queries
    INDEX idx_timestamp (timestamp),
    INDEX idx_class (class_label),
    INDEX idx_created (created_at)
);

-- Training runs history
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp DATETIME NOT NULL,
    model_path TEXT NOT NULL,

    -- Training data range
    data_start_date DATETIME,
    data_end_date DATETIME,
    n_training_samples INTEGER,
    n_validation_samples INTEGER,

    -- Performance metrics
    train_accuracy REAL,
    val_accuracy REAL,
    test_accuracy REAL,  -- On permanent test set
    roc_auc REAL,

    -- Training metadata
    update_mode TEXT,  -- 'incremental' or 'full_retrain'
    parent_model_id INTEGER,  -- For incremental updates
    config_snapshot TEXT,  -- JSON
    duration_seconds REAL,

    FOREIGN KEY (parent_model_id) REFERENCES training_runs(id)
);

-- Weekly validation metrics
CREATE TABLE validation_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    val_timestamp DATETIME NOT NULL,
    training_run_id INTEGER NOT NULL,

    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    roc_auc REAL,

    -- Confusion matrix
    true_negatives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    true_positives INTEGER,

    FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
);

-- Data drift tracking
CREATE TABLE drift_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_timestamp DATETIME NOT NULL,

    -- Date range analyzed
    analysis_start DATETIME,
    analysis_end DATETIME,

    -- Drift scores
    kl_divergence REAL,  -- Overall drift from baseline
    mean_shift_max REAL,  -- Max feature mean shift
    std_shift_max REAL,   -- Max feature std shift

    -- Class distribution
    class_0_percentage REAL,
    class_1_percentage REAL,

    -- Feature statistics (JSON arrays)
    feature_means TEXT,
    feature_stds TEXT
);
```

**Total database size** for common scenarios:

| Duration | Samples (stereo) | Database Size | % of 2TB |
|----------|------------------|---------------|----------|
| 1 month (24/7) | 5.18M | 34.6 GB | 1.7% |
| 6 months | 31.1M | 207 GB | 10.4% |
| 1 year | 62.2M | 415 GB | 20.8% |
| 3 years | 186.6M | 1.24 TB | 62.0% |
| 5 years | 307M | 2.05 TB | 100% |

---

## Implementation

### 1. Stereo Channel Separator

**File**: `src/continuous/stereo_channel_processor.py`

```python
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class StereoChannelProcessor:
    """
    Process stereo WAV files where:
    - Left channel (0) = Positive class samples
    - Right channel (1) = Negative class samples
    """

    def __init__(self, config):
        self.sample_rate = config.audio.sample_rate
        self.segment_duration = config.audio.segment_duration
        self.n_mfcc = config.audio.n_mfcc
        self.feature_types = config.audio.feature_types

    def process_stereo_file(self, wav_path: Path,
                           max_samples_per_channel: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process stereo WAV file and extract features from both channels.

        Args:
            wav_path: Path to stereo WAV file
            max_samples_per_channel: Maximum segments to extract per channel (None = all)

        Returns:
            X_left: Features from left channel (positive class)
            y_left: Labels for left channel (all 1s)
            X_right: Features from right channel (negative class)
            y_right: Labels for right channel (all 0s)
        """
        logger.info(f"Processing stereo file: {wav_path}")

        # Load stereo audio
        audio, sr = librosa.load(wav_path, sr=self.sample_rate, mono=False)

        # Check if actually stereo
        if len(audio.shape) == 1:
            raise ValueError(f"Expected stereo audio, got mono: {wav_path}")

        if audio.shape[0] != 2:
            raise ValueError(f"Expected 2 channels, got {audio.shape[0]}: {wav_path}")

        # Separate channels
        left_channel = audio[0, :]   # Channel 0 = positive
        right_channel = audio[1, :]  # Channel 1 = negative

        logger.info(f"Duration: {len(left_channel)/sr:.1f} seconds")

        # Extract features from each channel
        X_left, offsets_left = self._extract_features_from_channel(
            left_channel, sr, max_samples_per_channel
        )
        X_right, offsets_right = self._extract_features_from_channel(
            right_channel, sr, max_samples_per_channel
        )

        # Create labels
        y_left = np.ones(len(X_left), dtype=int)   # Positive class
        y_right = np.zeros(len(X_right), dtype=int)  # Negative class

        logger.info(f"Extracted {len(X_left)} positive samples, {len(X_right)} negative samples")

        return X_left, y_left, X_right, y_right, offsets_left, offsets_right

    def _extract_features_from_channel(self, channel_audio: np.ndarray,
                                      sr: int,
                                      max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from single channel.

        Returns:
            features: (n_samples, n_features) array
            offsets: (n_samples,) array of segment start times in seconds
        """
        segment_length = int(self.segment_duration * sr)
        total_samples = len(channel_audio)

        # Calculate number of possible segments
        n_possible_segments = total_samples // segment_length

        # Limit if requested
        if max_samples is not None:
            n_segments = min(n_possible_segments, max_samples)
        else:
            n_segments = n_possible_segments

        # For maximum data capture, use sequential segments (not random)
        # This ensures we use every second of audio
        features_list = []
        offsets_list = []

        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length

            segment = channel_audio[start_idx:end_idx]

            # Extract features
            features = self._extract_features_from_segment(segment, sr)
            features_list.append(features)

            # Store offset in seconds
            offset_sec = start_idx / sr
            offsets_list.append(offset_sec)

        return np.array(features_list), np.array(offsets_list)

    def _extract_features_from_segment(self, segment: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract features from 1-second audio segment.

        Returns:
            features: (680,) array for default config
        """
        features = []

        # MFCC
        if 'mfcc' in self.feature_types:
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=self.n_mfcc)
            features.append(mfcc.flatten())

        # Spectral features
        if 'spectral' in self.feature_types:
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(segment)

            features.extend([
                spectral_centroid.flatten(),
                spectral_rolloff.flatten(),
                spectral_bandwidth.flatten(),
                zero_crossing_rate.flatten()
            ])

        # Chroma (optional)
        if 'chroma' in self.feature_types:
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            features.append(chroma.flatten())

        # Tonnetz (optional)
        if 'tonnetz' in self.feature_types:
            tonnetz = librosa.feature.tonnetz(y=segment, sr=sr)
            features.append(tonnetz.flatten())

        # Concatenate all features
        return np.concatenate(features)


def simulate_continuous_stream(source_wav: Path,
                               chunk_duration_minutes: int,
                               config) -> List[Tuple[Path, float]]:
    """
    Simulate continuous data stream by splitting a large WAV file into chunks.

    This simulates what would happen with real-time recording:
    - Every N minutes, a new chunk is "recorded"
    - Chunks are processed as they arrive

    Args:
        source_wav: Large stereo WAV file to split
        chunk_duration_minutes: Duration of each chunk
        config: Audio configuration

    Returns:
        List of (chunk_path, timestamp) tuples
    """
    import tempfile
    import soundfile as sf
    from datetime import datetime, timedelta

    logger.info(f"Simulating continuous stream from {source_wav}")

    # Load full audio
    audio, sr = librosa.load(source_wav, sr=config.audio.sample_rate, mono=False)

    total_duration_sec = audio.shape[1] / sr
    chunk_duration_sec = chunk_duration_minutes * 60

    n_chunks = int(total_duration_sec / chunk_duration_sec)

    logger.info(f"Creating {n_chunks} chunks of {chunk_duration_minutes} min each")

    # Create temporary directory for chunks
    temp_dir = Path(tempfile.mkdtemp(prefix='continuous_stream_'))

    chunks = []
    start_time = datetime.now()

    for i in range(n_chunks):
        # Extract chunk
        start_sample = int(i * chunk_duration_sec * sr)
        end_sample = int((i + 1) * chunk_duration_sec * sr)

        chunk_audio = audio[:, start_sample:end_sample]

        # Save chunk
        chunk_path = temp_dir / f"chunk_{i:04d}.wav"
        sf.write(chunk_path, chunk_audio.T, sr)  # Transpose for soundfile

        # Simulate timestamp (chunks arriving every N minutes)
        chunk_timestamp = start_time + timedelta(minutes=i * chunk_duration_minutes)

        chunks.append((chunk_path, chunk_timestamp))

        logger.debug(f"Created chunk {i+1}/{n_chunks}: {chunk_path}")

    return chunks
```

### 2. Feature Database Manager

**File**: `src/continuous/feature_database.py`

```python
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import logging
import io

logger = logging.getLogger(__name__)


class FeatureDatabase:
    """
    Manage feature storage in SQLite database.
    Stores only transformed features (not raw audio).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Features table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                class_label INTEGER NOT NULL,
                channel INTEGER NOT NULL,
                source_file TEXT,
                segment_offset_sec REAL,
                features BLOB NOT NULL,
                extraction_version TEXT DEFAULT 'v1.0',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Indexes for fast queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON features(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_class ON features(class_label)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created ON features(created_at)')

        # Training runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_timestamp DATETIME NOT NULL,
                model_path TEXT NOT NULL,
                data_start_date DATETIME,
                data_end_date DATETIME,
                n_training_samples INTEGER,
                n_validation_samples INTEGER,
                train_accuracy REAL,
                val_accuracy REAL,
                test_accuracy REAL,
                roc_auc REAL,
                update_mode TEXT,
                parent_model_id INTEGER,
                config_snapshot TEXT,
                duration_seconds REAL,
                FOREIGN KEY (parent_model_id) REFERENCES training_runs(id)
            )
        ''')

        # Validation metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS validation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                val_timestamp DATETIME NOT NULL,
                training_run_id INTEGER NOT NULL,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                true_negatives INTEGER,
                false_positives INTEGER,
                false_negatives INTEGER,
                true_positives INTEGER,
                FOREIGN KEY (training_run_id) REFERENCES training_runs(id)
            )
        ''')

        # Drift metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_timestamp DATETIME NOT NULL,
                analysis_start DATETIME,
                analysis_end DATETIME,
                kl_divergence REAL,
                mean_shift_max REAL,
                std_shift_max REAL,
                class_0_percentage REAL,
                class_1_percentage REAL,
                feature_means TEXT,
                feature_stds TEXT
            )
        ''')

        conn.commit()
        conn.close()

        logger.info(f"Database initialized: {self.db_path}")

    def insert_features_batch(self,
                             X: np.ndarray,
                             y: np.ndarray,
                             channel: int,
                             source_file: str,
                             offsets: np.ndarray,
                             timestamp: datetime = None):
        """
        Insert batch of features into database.

        Args:
            X: (n_samples, n_features) feature array
            y: (n_samples,) label array
            channel: 0 or 1 (right or left)
            source_file: Original WAV filename
            offsets: (n_samples,) segment offsets in seconds
            timestamp: Recording timestamp (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Insert each sample
        for i in range(len(X)):
            # Compress features to binary
            features_blob = self._numpy_to_blob(X[i])

            cursor.execute('''
                INSERT INTO features
                (timestamp, class_label, channel, source_file, segment_offset_sec, features)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                int(y[i]),
                channel,
                source_file,
                float(offsets[i]),
                features_blob
            ))

        conn.commit()
        conn.close()

        logger.info(f"Inserted {len(X)} features from {source_file}")

    def load_features_by_date_range(self,
                                    start_date: datetime,
                                    end_date: datetime,
                                    max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features from database by date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            max_samples: Maximum samples to load (None = all)

        Returns:
            X: (n_samples, n_features) feature array
            y: (n_samples,) label array
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if max_samples is None:
            cursor.execute('''
                SELECT features, class_label
                FROM features
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_date, end_date))
        else:
            cursor.execute('''
                SELECT features, class_label
                FROM features
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
                LIMIT ?
            ''', (start_date, end_date, max_samples))

        rows = cursor.fetchall()
        conn.close()

        if len(rows) == 0:
            logger.warning(f"No features found for range {start_date} to {end_date}")
            return np.array([]), np.array([])

        # Decompress features
        X_list = []
        y_list = []

        for features_blob, label in rows:
            features = self._blob_to_numpy(features_blob)
            X_list.append(features)
            y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Loaded {len(X)} features from {start_date} to {end_date}")

        return X, y

    def get_database_stats(self) -> dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total samples
        cursor.execute('SELECT COUNT(*) FROM features')
        total_samples = cursor.fetchone()[0]

        # Class distribution
        cursor.execute('''
            SELECT class_label, COUNT(*)
            FROM features
            GROUP BY class_label
        ''')
        class_counts = dict(cursor.fetchall())

        # Date range
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM features')
        date_min, date_max = cursor.fetchone()

        # Database file size
        db_size_bytes = self.db_path.stat().st_size

        conn.close()

        return {
            'total_samples': total_samples,
            'class_0_count': class_counts.get(0, 0),
            'class_1_count': class_counts.get(1, 0),
            'date_range_start': date_min,
            'date_range_end': date_max,
            'db_size_mb': db_size_bytes / (1024 * 1024),
            'db_size_gb': db_size_bytes / (1024 * 1024 * 1024)
        }

    @staticmethod
    def _numpy_to_blob(array: np.ndarray) -> bytes:
        """Convert numpy array to compressed binary blob"""
        out = io.BytesIO()
        np.savez_compressed(out, data=array)
        return out.getvalue()

    @staticmethod
    def _blob_to_numpy(blob: bytes) -> np.ndarray:
        """Convert binary blob to numpy array"""
        in_bytes = io.BytesIO(blob)
        return np.load(in_bytes, allow_pickle=False)['data']

    def insert_training_run(self, run_info: dict) -> int:
        """Insert training run record and return run_id"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO training_runs
            (run_timestamp, model_path, data_start_date, data_end_date,
             n_training_samples, n_validation_samples, train_accuracy, val_accuracy,
             test_accuracy, roc_auc, update_mode, parent_model_id, config_snapshot,
             duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_info['run_timestamp'],
            run_info['model_path'],
            run_info.get('data_start_date'),
            run_info.get('data_end_date'),
            run_info.get('n_training_samples'),
            run_info.get('n_validation_samples'),
            run_info.get('train_accuracy'),
            run_info.get('val_accuracy'),
            run_info.get('test_accuracy'),
            run_info.get('roc_auc'),
            run_info.get('update_mode', 'incremental'),
            run_info.get('parent_model_id'),
            run_info.get('config_snapshot'),
            run_info.get('duration_seconds')
        ))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return run_id
```

This is getting quite long. Let me create the remaining key files and then commit. Should I continue with:
1. Continuous ingestion orchestrator
2. Test suite with real stereo data
3. Configuration files

Or would you like me to commit what we have so far with full documentation?