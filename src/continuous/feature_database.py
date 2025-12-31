"""
Feature Database Manager for Continuous Learning

Manages SQLite database storing transformed audio features (not raw audio).
Optimized for 2TB storage with efficient compressed feature storage.
"""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import logging
import io
import json

logger = logging.getLogger(__name__)


class FeatureDatabase:
    """
    Manage feature storage in SQLite database.

    Database design:
    - Stores only transformed features (not raw audio)
    - Features compressed as numpy arrays in BLOBs
    - Indexed for fast temporal queries
    - Tracks training history and performance metrics
    """

    def __init__(self, db_path: Path):
        """
        Initialize database connection and schema.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

        logger.info(f"FeatureDatabase initialized: {self.db_path}")

    def _init_database(self):
        """Initialize database schema if not exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Features table - main storage
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON features(source_file)')

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

        logger.debug(f"Database schema initialized")

    def insert_features_batch(self,
                              X: np.ndarray,
                              y: np.ndarray,
                              channel: int,
                              source_file: str,
                              offsets: np.ndarray,
                              timestamp: Optional[datetime] = None):
        """
        Insert batch of features into database.

        Args:
            X: (n_samples, n_features) feature array
            y: (n_samples,) label array
            channel: 0 (right/negative) or 1 (left/positive)
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

        logger.info(f"Inserted {len(X)} features from {source_file} (channel {channel})")

    def load_features_by_date_range(self,
                                     start_date: datetime,
                                     end_date: datetime,
                                     max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load features from database by date range.

        Args:
            start_date: Start of date range
            end_date: End of date range
            max_samples: Maximum samples to load (None = all)

        Returns:
            Tuple of (X, y)
            - X: (n_samples, n_features) feature array
            - y: (n_samples,) label array
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

    def get_database_stats(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary with database metrics
        """
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
        db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0

        conn.close()

        stats = {
            'total_samples': total_samples,
            'class_0_count': class_counts.get(0, 0),
            'class_1_count': class_counts.get(1, 0),
            'date_range_start': date_min,
            'date_range_end': date_max,
            'db_size_mb': db_size_bytes / (1024 * 1024),
            'db_size_gb': db_size_bytes / (1024 * 1024 * 1024),
            'class_balance': class_counts.get(1, 0) / total_samples if total_samples > 0 else 0.0
        }

        return stats

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

    def insert_training_run(self, run_info: Dict) -> int:
        """
        Insert training run record and return run_id.

        Args:
            run_info: Dictionary with training run metadata

        Returns:
            Training run ID
        """
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
            json.dumps(run_info.get('config_snapshot')) if run_info.get('config_snapshot') else None,
            run_info.get('duration_seconds')
        ))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Inserted training run {run_id}")

        return run_id

    def insert_validation_metrics(self, metrics: Dict):
        """
        Insert validation metrics record.

        Args:
            metrics: Dictionary with validation metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO validation_metrics
            (val_timestamp, training_run_id, accuracy, precision, recall, f1_score,
             roc_auc, true_negatives, false_positives, false_negatives, true_positives)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['val_timestamp'],
            metrics['training_run_id'],
            metrics.get('accuracy'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('f1_score'),
            metrics.get('roc_auc'),
            metrics.get('true_negatives'),
            metrics.get('false_positives'),
            metrics.get('false_negatives'),
            metrics.get('true_positives')
        ))

        conn.commit()
        conn.close()

        logger.info(f"Inserted validation metrics for run {metrics['training_run_id']}")

    def insert_drift_metrics(self, metrics: Dict):
        """
        Insert drift metrics record.

        Args:
            metrics: Dictionary with drift metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO drift_metrics
            (metric_timestamp, analysis_start, analysis_end, kl_divergence,
             mean_shift_max, std_shift_max, class_0_percentage, class_1_percentage,
             feature_means, feature_stds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['metric_timestamp'],
            metrics.get('analysis_start'),
            metrics.get('analysis_end'),
            metrics.get('kl_divergence'),
            metrics.get('mean_shift_max'),
            metrics.get('std_shift_max'),
            metrics.get('class_0_percentage'),
            metrics.get('class_1_percentage'),
            json.dumps(metrics.get('feature_means').tolist()) if 'feature_means' in metrics else None,
            json.dumps(metrics.get('feature_stds').tolist()) if 'feature_stds' in metrics else None
        ))

        conn.commit()
        conn.close()

        logger.info(f"Inserted drift metrics for {metrics['metric_timestamp']}")

    def get_latest_training_run(self) -> Optional[Dict]:
        """
        Get the most recent training run.

        Returns:
            Dictionary with training run info, or None if no runs exist
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, run_timestamp, model_path, update_mode, test_accuracy, roc_auc
            FROM training_runs
            ORDER BY run_timestamp DESC
            LIMIT 1
        ''')

        row = cursor.fetchone()
        conn.close()

        if row is None:
            return None

        return {
            'id': row[0],
            'run_timestamp': row[1],
            'model_path': row[2],
            'update_mode': row[3],
            'test_accuracy': row[4],
            'roc_auc': row[5]
        }
