#!/usr/bin/env python3
"""
Assign random class labels to both channels of stereo recordings.

For each recording, one channel gets class 0 and the other gets class 1.
Which channel gets which class is randomly determined.

This ensures balanced class distribution across the dataset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import random
import logging
from datetime import datetime
import yaml

from continuous.stereo_channel_processor import StereoChannelProcessor
from continuous.feature_database import FeatureDatabase
from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    # Configuration
    config_path = Path('config/continuous_learning_config.yaml')
    db_path = Path('data/continuous/test_24hr.db')
    recordings_dir = Path('data/continuous/recordings')

    # Load config
    config = load_config(str(config_path))

    # Initialize processor and database
    processor = StereoChannelProcessor(config)
    db = FeatureDatabase(db_path=db_path, backend='sqlite')

    # Get list of WAV files (exclude already processed test_recording)
    wav_files = sorted(recordings_dir.glob('*.wav'))

    # Check which files are already in the database
    already_processed = set()
    with db.db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT source_file FROM features")
        for row in cursor.fetchall():
            already_processed.add(row[0])

    logger.info(f"Found {len(wav_files)} WAV files")
    logger.info(f"Already processed: {already_processed}")

    # Filter to only unprocessed files
    files_to_process = [f for f in wav_files if f.name not in already_processed]
    logger.info(f"Files to process: {len(files_to_process)}")

    # Set random seed for reproducibility
    random.seed(42)

    # Process each file with random class assignments
    # Each recording gets one channel as class 0 and the other as class 1
    results = []
    for wav_file in files_to_process:
        # Randomly decide which channel gets which class
        # If left_class is 0, right_class is 1, and vice versa
        left_class = random.randint(0, 1)
        right_class = 1 - left_class  # Opposite class

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {wav_file.name}")
        logger.info(f"  Left channel (0) -> class {left_class}")
        logger.info(f"  Right channel (1) -> class {right_class}")

        try:
            # Process stereo file with custom class assignments
            # Note: positive_label is for LEFT channel, negative_label is for RIGHT channel
            X_left, y_left, X_right, y_right, offsets_left, offsets_right = \
                processor.process_stereo_file(
                    wav_file,
                    max_samples_per_channel=1000,  # Limit samples per channel
                    positive_label=left_class,     # Left channel class
                    negative_label=right_class     # Right channel class
                )

            # Extract timestamp from filename if possible
            # Format: continuous_expexp_4310cb8d_cycle110_2026-01-09_07-42-55_stereo.wav
            try:
                parts = wav_file.stem.split('_')
                # Find date part (YYYY-MM-DD)
                date_idx = None
                for i, part in enumerate(parts):
                    if len(part) == 10 and part[4] == '-' and part[7] == '-':
                        date_idx = i
                        break

                if date_idx and date_idx + 1 < len(parts):
                    date_str = parts[date_idx]
                    time_str = parts[date_idx + 1]
                    timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y-%m-%d_%H-%M-%S")
                else:
                    timestamp = datetime.now()
            except Exception:
                timestamp = datetime.now()

            logger.info(f"  Using timestamp: {timestamp}")

            # Store in database
            # Channel 0 = left channel
            db.insert_features_batch(
                X=X_left,
                y=y_left,
                channel=0,
                source_file=wav_file.name,
                offsets=offsets_left,
                timestamp=timestamp
            )

            # Channel 1 = right channel
            db.insert_features_batch(
                X=X_right,
                y=y_right,
                channel=1,
                source_file=wav_file.name,
                offsets=offsets_right,
                timestamp=timestamp
            )

            results.append({
                'file': wav_file.name,
                'left_class': left_class,
                'right_class': right_class,
                'left_samples': len(X_left),
                'right_samples': len(X_right),
                'timestamp': timestamp
            })

            logger.info(f"  Stored {len(X_left)} left + {len(X_right)} right samples")

        except Exception as e:
            logger.error(f"  ERROR processing {wav_file.name}: {e}")
            continue

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - Random Class Assignments")
    print("="*80)
    print(f"{'File':<60} {'Left':>6} {'Right':>6} {'Samples':>10}")
    print("-"*80)

    for r in results:
        samples = f"{r['left_samples']}+{r['right_samples']}"
        print(f"{r['file']:<60} {r['left_class']:>6} {r['right_class']:>6} {samples:>10}")

    print("-"*80)
    print(f"Total files processed: {len(results)}")

    # Show class distribution
    total_class_0 = sum(r['left_samples'] for r in results if r['left_class'] == 0)
    total_class_0 += sum(r['right_samples'] for r in results if r['right_class'] == 0)
    total_class_1 = sum(r['left_samples'] for r in results if r['left_class'] == 1)
    total_class_1 += sum(r['right_samples'] for r in results if r['right_class'] == 1)

    print(f"\nClass distribution:")
    print(f"  Class 0: {total_class_0} samples")
    print(f"  Class 1: {total_class_1} samples")


if __name__ == '__main__':
    main()
