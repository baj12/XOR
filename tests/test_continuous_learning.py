"""
Test Suite for Continuous Learning Modules

Tests with REAL stereo WAV files from data/ directory.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Add src directory to path
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from continuous.stereo_channel_processor import StereoChannelProcessor, simulate_continuous_stream
from continuous.feature_database import FeatureDatabase


class MockAudioConfig:
    """Mock configuration for testing"""
    sample_rate = 22050  # Downsample for faster testing
    segment_duration = 1.0
    n_mfcc = 13
    feature_types = ['mfcc', 'spectral']


class MockConfig:
    def __init__(self):
        self.audio = MockAudioConfig()


class TestStereoChannelProcessor(unittest.TestCase):
    """Test stereo WAV file processing"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.config = MockConfig()
        cls.processor = StereoChannelProcessor(cls.config)

        # Find a real stereo WAV file in data directory
        data_dir = Path(__file__).parent.parent / 'data'

        # Try to find stereo WAV files
        wav_files = list(data_dir.glob('*.wav'))

        if len(wav_files) == 0:
            raise unittest.SkipTest("No WAV files found in data/ directory")

        # Use first WAV file
        cls.test_wav = wav_files[0]
        print(f"\nUsing test file: {cls.test_wav}")

    def test_process_stereo_file(self):
        """Test processing of real stereo WAV file"""
        # Process with limited samples for speed
        X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
            self.processor.process_stereo_file(self.test_wav, max_samples_per_channel=10)

        # Verify shapes
        self.assertEqual(X_left.shape[0], 10, "Should extract 10 left channel samples")
        self.assertEqual(X_right.shape[0], 10, "Should extract 10 right channel samples")

        # Verify feature dimensions
        # Actual dimensions depend on librosa's frame calculation
        # For 1-sec at 22050 Hz: varies based on hop_length/n_fft
        # Just verify it's in reasonable range and consistent
        self.assertGreater(X_left.shape[1], 600, "Should have at least 600 features")
        self.assertLess(X_left.shape[1], 900, "Should have less than 900 features")
        self.assertEqual(X_right.shape[1], X_left.shape[1], "Both channels should have same features")

        # Verify labels
        self.assertTrue(np.all(y_left == 1), "Left channel should have label 1 (positive)")
        self.assertTrue(np.all(y_right == 0), "Right channel should have label 0 (negative)")

        # Verify offsets
        self.assertEqual(len(offsets_left), 10, "Should have offsets for each sample")
        self.assertTrue(np.all(np.diff(offsets_left) > 0), "Offsets should be increasing")

        print(f"✓ Processed stereo file: {X_left.shape[0]} + {X_right.shape[0]} samples")
        print(f"✓ Feature dimensions: {X_left.shape[1]}D")

    def test_feature_extraction_quality(self):
        """Test that extracted features are valid"""
        X_left, _, X_right, _, _, _ = \
            self.processor.process_stereo_file(self.test_wav, max_samples_per_channel=5)

        # Features should not be all zeros
        self.assertFalse(np.all(X_left == 0), "Features should not be all zero")
        self.assertFalse(np.all(X_right == 0), "Features should not be all zero")

        # Features should not have NaN or Inf
        self.assertFalse(np.any(np.isnan(X_left)), "Features should not contain NaN")
        self.assertFalse(np.any(np.isinf(X_left)), "Features should not contain Inf")

        # Features should have reasonable range
        # Audio features (especially MFCC) can have large values before normalization
        self.assertTrue(np.abs(X_left).max() < 10000, "Features should have reasonable magnitude")
        self.assertTrue(np.abs(X_right).max() < 10000, "Features should have reasonable magnitude")

        print(f"✓ Feature quality checks passed")
        print(f"  Left channel: mean={X_left.mean():.2f}, std={X_left.std():.2f}")
        print(f"  Right channel: mean={X_right.mean():.2f}, std={X_right.std():.2f}")

    def test_simulate_continuous_stream(self):
        """Test simulation of continuous data stream"""
        # Create 5-minute chunks from test file
        chunks = simulate_continuous_stream(
            self.test_wav,
            5,
            self.config
        )

        # Should create at least 1 chunk
        self.assertGreater(len(chunks), 0, "Should create at least one chunk")

        # Each chunk should be a tuple of (path, timestamp)
        for chunk_path, timestamp in chunks:
            self.assertTrue(chunk_path.exists(), f"Chunk file should exist: {chunk_path}")
            self.assertIsInstance(timestamp, datetime, "Timestamp should be datetime")

        # Clean up
        if len(chunks) > 0:
            temp_dir = chunks[0][0].parent
            shutil.rmtree(temp_dir)

        print(f"✓ Simulated {len(chunks)} chunks")


class TestFeatureDatabase(unittest.TestCase):
    """Test feature database operations"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.temp_dir = tempfile.mkdtemp(prefix='test_db_')
        cls.db_path = Path(cls.temp_dir) / 'test_features.db'
        cls.db = FeatureDatabase(cls.db_path)

        print(f"\nTest database: {cls.db_path}")

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.temp_dir)

    def test_database_initialization(self):
        """Test database schema creation"""
        self.assertTrue(self.db_path.exists(), "Database file should be created")

        # Check tables exist
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        conn.close()

        expected_tables = ['features', 'training_runs', 'validation_metrics', 'drift_metrics']
        for table in expected_tables:
            self.assertIn(table, tables, f"Table {table} should exist")

        print(f"✓ Database schema created with {len(tables)} tables")

    def test_insert_and_load_features(self):
        """Test feature insertion and retrieval"""
        # Create mock features
        n_samples = 100
        n_features = 680
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)
        offsets = np.arange(n_samples, dtype=np.float32)

        # Use a unique timestamp far in the past to avoid overlap with other tests
        timestamp = datetime(2020, 1, 1, 12, 0, 0)

        # Insert features
        self.db.insert_features_batch(
            X, y, channel=0, source_file='test.wav',
            offsets=offsets, timestamp=timestamp
        )

        # Load features back with narrow date range around this timestamp
        start_date = timestamp - timedelta(hours=1)
        end_date = timestamp + timedelta(hours=1)

        X_loaded, y_loaded = self.db.load_features_by_date_range(start_date, end_date)

        # Verify - should only load samples from this test
        self.assertEqual(X_loaded.shape[0], n_samples, "Should load all samples")
        self.assertEqual(X_loaded.shape[1], n_features, "Should preserve feature dimensions")

        # Features should match (within compression tolerance)
        np.testing.assert_allclose(X_loaded, X, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(y_loaded, y)

        print(f"✓ Inserted and loaded {n_samples} samples")
        print(f"✓ Feature compression preserved accuracy")

    def test_database_stats(self):
        """Test database statistics calculation"""
        # First insert some data to ensure stats are meaningful
        n_samples = 50
        n_features = 680
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randint(0, 2, n_samples)
        offsets = np.arange(n_samples, dtype=np.float32)

        self.db.insert_features_batch(
            X, y, channel=0, source_file='test_stats.wav',
            offsets=offsets, timestamp=datetime.now()
        )

        stats = self.db.get_database_stats()

        # Should have stats
        self.assertGreater(stats['total_samples'], 0, "Should have samples")
        self.assertIn('class_0_count', stats)
        self.assertIn('class_1_count', stats)
        self.assertIn('db_size_mb', stats)

        print(f"✓ Database stats:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Class 0: {stats['class_0_count']}, Class 1: {stats['class_1_count']}")
        print(f"  Size: {stats['db_size_mb']:.2f} MB")

    def test_training_run_tracking(self):
        """Test training run metadata storage"""
        run_info = {
            'run_timestamp': datetime.now(),
            'model_path': '/path/to/model.keras',
            'data_start_date': datetime.now() - timedelta(days=7),
            'data_end_date': datetime.now(),
            'n_training_samples': 1000,
            'n_validation_samples': 200,
            'train_accuracy': 0.95,
            'val_accuracy': 0.92,
            'test_accuracy': 0.91,
            'roc_auc': 0.96,
            'update_mode': 'incremental',
            'duration_seconds': 120.5
        }

        run_id = self.db.insert_training_run(run_info)

        self.assertIsInstance(run_id, int, "Should return run ID")
        self.assertGreater(run_id, 0, "Run ID should be positive")

        # Retrieve latest run
        latest_run = self.db.get_latest_training_run()
        self.assertIsNotNone(latest_run)
        self.assertEqual(latest_run['id'], run_id)

        print(f"✓ Inserted training run {run_id}")
        print(f"  Accuracy: {latest_run['test_accuracy']:.2%}")


class TestIntegration(unittest.TestCase):
    """Integration tests with real data"""

    @classmethod
    def setUpClass(cls):
        """Set up integration test"""
        cls.config = MockConfig()
        cls.processor = StereoChannelProcessor(cls.config)

        # Find real WAV file
        data_dir = Path(__file__).parent.parent / 'data'
        wav_files = list(data_dir.glob('*.wav'))

        if len(wav_files) == 0:
            raise unittest.SkipTest("No WAV files found")

        cls.test_wav = wav_files[0]

        # Create temp database
        cls.temp_dir = tempfile.mkdtemp(prefix='test_integration_')
        cls.db_path = Path(cls.temp_dir) / 'integration.db'
        cls.db = FeatureDatabase(cls.db_path)

        print(f"\nIntegration test using: {cls.test_wav}")

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.temp_dir)

    def test_full_pipeline(self):
        """Test complete pipeline: WAV → Features → Database → Retrieval"""
        # 1. Process stereo WAV file
        X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
            self.processor.process_stereo_file(self.test_wav, max_samples_per_channel=20)

        # 2. Store in database
        timestamp = datetime.now()

        self.db.insert_features_batch(
            X_left, y_left, channel=0, source_file=str(self.test_wav),
            offsets=offsets_left, timestamp=timestamp
        )

        self.db.insert_features_batch(
            X_right, y_right, channel=1, source_file=str(self.test_wav),
            offsets=offsets_right, timestamp=timestamp
        )

        # 3. Retrieve from database
        start_date = timestamp - timedelta(hours=1)
        end_date = timestamp + timedelta(hours=1)

        X_loaded, y_loaded = self.db.load_features_by_date_range(start_date, end_date)

        # 4. Verify
        total_samples = len(X_left) + len(X_right)
        self.assertEqual(len(X_loaded), total_samples, "Should load all samples")

        # Check class distribution
        n_positive = np.sum(y_loaded == 1)
        n_negative = np.sum(y_loaded == 0)

        self.assertEqual(n_positive, len(y_left), "Should have correct positive samples")
        self.assertEqual(n_negative, len(y_right), "Should have correct negative samples")

        # 5. Get stats
        stats = self.db.get_database_stats()

        print(f"✓ Full pipeline test passed")
        print(f"  Processed: {total_samples} samples")
        print(f"  Positive: {n_positive}, Negative: {n_negative}")
        print(f"  Database: {stats['db_size_mb']:.2f} MB")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)
