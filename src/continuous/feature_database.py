"""
Feature Database Manager for Continuous Learning

Manages database storing transformed audio features (not raw audio).
Supports both SQLite (local/dev) and MariaDB (production) backends.
Optimized for 2TB storage with efficient compressed feature storage.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional, Dict
import logging
import io
import json

from db_connection import DatabaseConnection

logger = logging.getLogger(__name__)


class FeatureDatabase:
    """
    Manage feature storage with support for SQLite and MariaDB backends.

    Database design:
    - Stores only transformed features (not raw audio)
    - Features compressed as numpy arrays in BLOBs
    - Indexed for fast temporal queries
    - Tracks training history and performance metrics

    Usage:
        # SQLite (default for local/dev)
        db = FeatureDatabase(db_path='features.db', backend='sqlite')

        # MariaDB (for production)
        db = FeatureDatabase(backend='mariadb')
    """

    def __init__(self,
                 db_path: Optional[Path] = None,
                 backend: str = 'sqlite'):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file (ignored for MariaDB)
            backend: 'sqlite' or 'mariadb'
        """
        self.backend = backend
        self.db_path = Path(db_path) if db_path else None

        if self.backend == 'sqlite' and not self.db_path:
            raise ValueError("db_path required for SQLite backend")

        if self.backend == 'sqlite':
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db = DatabaseConnection(backend=backend, db_path=db_path)

        # Test connection
        if not self.db.test_connection():
            raise RuntimeError(f"Failed to connect to {backend} database")

        # Initialize schema if needed
        self._initialize_schema()

        logger.info(f"FeatureDatabase initialized with {backend} backend")

    def _initialize_schema(self):
        """Create tables if they don't exist."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if self.backend == 'sqlite':
                # SQLite schema - create all tables for continuous learning

                # Features table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS features (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME NOT NULL,
                        class_label INTEGER NOT NULL,
                        channel INTEGER NOT NULL,
                        source_file TEXT NOT NULL,
                        segment_offset_sec REAL NOT NULL,
                        features BLOB NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_class ON features(class_label)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_channel ON features(channel)')

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
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (parent_model_id) REFERENCES training_runs(id) ON DELETE SET NULL
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_runs_timestamp ON training_runs(run_timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_training_runs_model_path ON training_runs(model_path)')

                # Validation metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS validation_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        val_timestamp DATETIME NOT NULL,
                        training_run_id INTEGER NOT NULL,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        roc_auc REAL,
                        true_negatives INTEGER,
                        false_positives INTEGER,
                        false_negatives INTEGER,
                        true_positives INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE CASCADE
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_metrics_timestamp ON validation_metrics(val_timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_metrics_training_run ON validation_metrics(training_run_id)')

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
                        feature_stds TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_drift_metrics_timestamp ON drift_metrics(metric_timestamp)')

            else:
                # MariaDB schema (tables should already exist from init_mariadb.py)
                # Just verify they exist
                cursor.execute("SHOW TABLES LIKE 'features'")
                if not cursor.fetchone():
                    logger.warning("features table does not exist in MariaDB. Run scripts/init_mariadb.py first.")

            conn.commit()
            logger.debug("Database schema initialized")

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

        ph = self.db.get_placeholder(self.backend)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Insert each sample
            for i in range(len(X)):
                # Compress features to binary
                features_blob = self._numpy_to_blob(X[i])

                sql = f'''
                    INSERT INTO features
                    (timestamp, class_label, channel, source_file, segment_offset_sec, features)
                    VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph})
                '''

                cursor.execute(sql, (
                    timestamp,
                    int(y[i]),
                    channel,
                    source_file,
                    float(offsets[i]),
                    features_blob
                ))

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
        ph = self.db.get_placeholder(self.backend)

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            if max_samples is None:
                sql = f'''
                    SELECT features, class_label
                    FROM features
                    WHERE timestamp BETWEEN {ph} AND {ph}
                    ORDER BY timestamp
                '''
                cursor.execute(sql, (start_date, end_date))
            else:
                sql = f'''
                    SELECT features, class_label
                    FROM features
                    WHERE timestamp BETWEEN {ph} AND {ph}
                    ORDER BY timestamp
                    LIMIT {ph}
                '''
                cursor.execute(sql, (start_date, end_date, max_samples))

            rows = cursor.fetchall()

        if len(rows) == 0:
            logger.warning(f"No features found for range {start_date} to {end_date}")
            return np.array([]), np.array([])

        # Decompress features
        X_list = []
        y_list = []

        for row in rows:
            features_blob = row[0]
            label = row[1]
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
        with self.db.get_connection() as conn:
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
            result = cursor.fetchone()
            date_min, date_max = result[0], result[1] if result else (None, None)

        # Database file size (SQLite only)
        if self.backend == 'sqlite' and self.db_path:
            db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        else:
            db_size_bytes = 0  # MariaDB: would need SHOW TABLE STATUS

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
        ph = self.db.get_placeholder(self.backend)

        sql = f'''
            INSERT INTO training_runs
            (run_timestamp, model_path, data_start_date, data_end_date,
             n_training_samples, n_validation_samples, train_accuracy, val_accuracy,
             test_accuracy, roc_auc, update_mode, parent_model_id, config_snapshot,
             duration_seconds)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        '''

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(sql, (
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

            # Get last insert ID
            if self.backend == 'sqlite':
                run_id = cursor.lastrowid
            else:
                run_id = cursor.lastrowid

        logger.info(f"Inserted training run {run_id}")

        return run_id

    def insert_validation_metrics(self, metrics: Dict):
        """
        Insert validation metrics record.

        Args:
            metrics: Dictionary with validation metrics
        """
        ph = self.db.get_placeholder(self.backend)

        sql = f'''
            INSERT INTO validation_metrics
            (val_timestamp, training_run_id, accuracy, precision_score, recall_score, f1_score,
             roc_auc, true_negatives, false_positives, false_negatives, true_positives)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        '''

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(sql, (
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

        logger.info(f"Inserted validation metrics for run {metrics['training_run_id']}")

    def insert_drift_metrics(self, metrics: Dict):
        """
        Insert drift metrics record.

        Args:
            metrics: Dictionary with drift metrics
        """
        ph = self.db.get_placeholder(self.backend)

        sql = f'''
            INSERT INTO drift_metrics
            (metric_timestamp, analysis_start, analysis_end, kl_divergence,
             mean_shift_max, std_shift_max, class_0_percentage, class_1_percentage,
             feature_means, feature_stds)
            VALUES ({ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph}, {ph})
        '''

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(sql, (
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

        logger.info(f"Inserted drift metrics for {metrics['metric_timestamp']}")

    def get_latest_training_run(self) -> Optional[Dict]:
        """
        Get the most recent training run.

        Returns:
            Dictionary with training run info, or None if no runs exist
        """
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, run_timestamp, model_path, update_mode, test_accuracy, roc_auc
                FROM training_runs
                ORDER BY run_timestamp DESC
                LIMIT 1
            ''')

            row = cursor.fetchone()

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
