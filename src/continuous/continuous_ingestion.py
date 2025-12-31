"""
Continuous Ingestion Pipeline

Orchestrates continuous data flow from stereo WAV streams into the feature database.

Architecture:
- Monitors for new stereo WAV files (simulated or real-time)
- Processes both channels (left=positive, right=negative)
- Extracts features and stores in database
- Tracks ingestion metrics and errors
- Runs autonomously 24/7

Usage:
    from continuous.continuous_ingestion import ContinuousIngestionPipeline

    pipeline = ContinuousIngestionPipeline(config, db_path)
    pipeline.ingest_file(wav_path)  # Single file
    pipeline.run_continuous(data_dir)  # Continuous monitoring
"""

import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import numpy as np

from .stereo_channel_processor import StereoChannelProcessor
from .feature_database import FeatureDatabase

logger = logging.getLogger(__name__)


class IngestionMetrics:
    """Track ingestion pipeline metrics"""

    def __init__(self):
        self.files_processed = 0
        self.total_samples_ingested = 0
        self.total_features_extracted = 0
        self.errors = []
        self.start_time = datetime.now()
        self.last_ingestion_time = None

    def record_success(self, n_samples: int, n_features: int):
        """Record successful ingestion"""
        self.files_processed += 1
        self.total_samples_ingested += n_samples
        self.total_features_extracted += n_features
        self.last_ingestion_time = datetime.now()

    def record_error(self, error_msg: str):
        """Record ingestion error"""
        self.errors.append({
            'timestamp': datetime.now(),
            'error': error_msg
        })

    def get_summary(self) -> Dict:
        """Get metrics summary"""
        elapsed = (datetime.now() - self.start_time).total_seconds()

        return {
            'files_processed': self.files_processed,
            'total_samples': self.total_samples_ingested,
            'total_features': self.total_features_extracted,
            'elapsed_seconds': elapsed,
            'samples_per_second': self.total_samples_ingested / elapsed if elapsed > 0 else 0,
            'last_ingestion': self.last_ingestion_time,
            'error_count': len(self.errors),
            'recent_errors': self.errors[-5:] if self.errors else []
        }


class ContinuousIngestionPipeline:
    """
    Continuous ingestion pipeline for stereo WAV streams.

    Handles:
    - Stereo channel processing (left=positive, right=negative)
    - Feature extraction (MFCC + spectral)
    - Database storage with compression
    - Error handling and retry logic
    - Metrics tracking
    """

    def __init__(self,
                 config,
                 db_path: Path,
                 max_samples_per_channel: Optional[int] = None):
        """
        Initialize continuous ingestion pipeline.

        Args:
            config: Configuration object with audio parameters
            db_path: Path to SQLite database
            max_samples_per_channel: Maximum samples to extract per channel per file
                                    (None = extract all possible samples)
        """
        self.config = config
        self.processor = StereoChannelProcessor(config)
        self.db = FeatureDatabase(db_path)
        self.max_samples_per_channel = max_samples_per_channel
        self.metrics = IngestionMetrics()

        logger.info(f"ContinuousIngestionPipeline initialized")
        logger.info(f"Database: {db_path}")
        logger.info(f"Max samples per channel: {max_samples_per_channel or 'unlimited'}")

    def ingest_file(self, wav_path: Path, timestamp: Optional[datetime] = None) -> Dict:
        """
        Ingest single stereo WAV file into database.

        Args:
            wav_path: Path to stereo WAV file
            timestamp: Recording timestamp (defaults to now)

        Returns:
            Dict with ingestion results and metrics
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"Ingesting: {wav_path}")

        try:
            # Process stereo file
            X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
                self.processor.process_stereo_file(
                    wav_path,
                    max_samples_per_channel=self.max_samples_per_channel
                )

            n_left = len(X_left)
            n_right = len(X_right)
            n_features = X_left.shape[1]

            logger.info(f"Extracted: {n_left} positive + {n_right} negative samples ({n_features}D)")

            # Store in database
            # Left channel (positive class)
            self.db.insert_features_batch(
                X_left, y_left,
                channel=0,
                source_file=str(wav_path),
                offsets=offsets_left,
                timestamp=timestamp
            )

            # Right channel (negative class)
            self.db.insert_features_batch(
                X_right, y_right,
                channel=1,
                source_file=str(wav_path),
                offsets=offsets_right,
                timestamp=timestamp
            )

            # Record metrics
            total_samples = n_left + n_right
            self.metrics.record_success(total_samples, n_features)

            result = {
                'status': 'success',
                'file': str(wav_path),
                'timestamp': timestamp,
                'samples_left': n_left,
                'samples_right': n_right,
                'total_samples': total_samples,
                'n_features': n_features
            }

            logger.info(f"✓ Ingested {total_samples} samples from {wav_path.name}")

            return result

        except Exception as e:
            error_msg = f"Failed to ingest {wav_path}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.metrics.record_error(error_msg)

            return {
                'status': 'error',
                'file': str(wav_path),
                'timestamp': timestamp,
                'error': str(e)
            }

    def ingest_batch(self, wav_files: List[Path], timestamps: Optional[List[datetime]] = None) -> List[Dict]:
        """
        Ingest batch of stereo WAV files.

        Args:
            wav_files: List of WAV file paths
            timestamps: Optional list of timestamps (one per file)

        Returns:
            List of ingestion results
        """
        if timestamps is None:
            timestamps = [None] * len(wav_files)

        results = []
        for wav_path, timestamp in zip(wav_files, timestamps):
            result = self.ingest_file(wav_path, timestamp)
            results.append(result)

        return results

    def run_continuous(self,
                      data_dir: Path,
                      check_interval_seconds: int = 60,
                      file_pattern: str = "*.wav",
                      stop_after_seconds: Optional[int] = None) -> None:
        """
        Run continuous ingestion monitoring a directory for new files.

        Args:
            data_dir: Directory to monitor for new WAV files
            check_interval_seconds: How often to check for new files
            file_pattern: Glob pattern for WAV files (e.g., "*.wav")
            stop_after_seconds: Optional timeout for testing (None = run forever)
        """
        logger.info(f"Starting continuous ingestion from {data_dir}")
        logger.info(f"Check interval: {check_interval_seconds}s")
        logger.info(f"File pattern: {file_pattern}")

        processed_files = set()
        start_time = time.time()

        while True:
            # Check for timeout (testing mode)
            if stop_after_seconds and (time.time() - start_time) > stop_after_seconds:
                logger.info(f"Stopping after {stop_after_seconds}s (timeout)")
                break

            # Find new files
            all_files = set(data_dir.glob(file_pattern))
            new_files = all_files - processed_files

            if new_files:
                logger.info(f"Found {len(new_files)} new file(s)")

                for wav_path in sorted(new_files):
                    # Ingest file
                    result = self.ingest_file(wav_path)

                    if result['status'] == 'success':
                        processed_files.add(wav_path)

                    # Log metrics
                    metrics = self.metrics.get_summary()
                    logger.info(f"Metrics: {metrics['files_processed']} files, "
                              f"{metrics['total_samples']} samples, "
                              f"{metrics['samples_per_second']:.1f} samples/sec")

            # Wait before next check
            time.sleep(check_interval_seconds)

    def get_metrics(self) -> Dict:
        """Get current ingestion metrics"""
        return self.metrics.get_summary()

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return self.db.get_database_stats()


class SimulatedContinuousStream:
    """
    Simulate continuous stream by chunking large stereo WAV files.

    Useful for testing the pipeline with existing data before deploying
    with real-time recording.
    """

    def __init__(self,
                 source_wav: Path,
                 output_dir: Path,
                 chunk_duration_minutes: int = 5):
        """
        Initialize simulated stream.

        Args:
            source_wav: Large stereo WAV file to chunk
            output_dir: Where to write chunks
            chunk_duration_minutes: Duration of each chunk
        """
        self.source_wav = source_wav
        self.output_dir = output_dir
        self.chunk_duration_minutes = chunk_duration_minutes
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Simulated stream: {source_wav}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Chunk size: {chunk_duration_minutes} minutes")

    def generate_chunks(self, config) -> List[Path]:
        """
        Generate chunks from source WAV file.

        Args:
            config: Audio configuration

        Returns:
            List of chunk file paths
        """
        from .stereo_channel_processor import simulate_continuous_stream

        logger.info("Generating chunks...")

        chunks = simulate_continuous_stream(
            self.source_wav,
            self.chunk_duration_minutes,
            config
        )

        # Move chunks to output directory
        chunk_paths = []
        for i, (chunk_path, timestamp) in enumerate(chunks):
            # Create timestamped filename
            new_name = f"chunk_{timestamp.strftime('%Y%m%d_%H%M%S')}.wav"
            new_path = self.output_dir / new_name

            # Move chunk
            chunk_path.rename(new_path)
            chunk_paths.append(new_path)

            logger.debug(f"Chunk {i+1}/{len(chunks)}: {new_path}")

        logger.info(f"✓ Generated {len(chunk_paths)} chunks")

        return chunk_paths

    def simulate_realtime_arrival(self,
                                  chunks: List[Path],
                                  arrival_interval_seconds: int = 300) -> None:
        """
        Simulate real-time chunk arrival by controlling file timestamps.

        Useful for testing continuous ingestion pipeline as if chunks
        are arriving in real-time.

        Args:
            chunks: List of chunk file paths
            arrival_interval_seconds: Simulated time between chunks
        """
        logger.info(f"Simulating real-time arrival ({arrival_interval_seconds}s intervals)")

        base_time = datetime.now()

        for i, chunk_path in enumerate(chunks):
            # Set file timestamp to simulate arrival time
            arrival_time = base_time + timedelta(seconds=i * arrival_interval_seconds)
            timestamp = arrival_time.timestamp()

            # Update file timestamps (atime, mtime)
            chunk_path.touch()
            import os
            os.utime(chunk_path, (timestamp, timestamp))

            logger.debug(f"Chunk {i+1}: simulated arrival at {arrival_time}")
