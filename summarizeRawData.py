import glob
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_quadrants(df):
    """Analyze data distribution and statistics across quadrants."""

    # Define quadrants based on x,y coordinates
    df['quadrant'] = df.apply(lambda row: (
        'Q1' if row['x'] >= 0 and row['y'] >= 0 else
        'Q2' if row['x'] < 0 and row['y'] >= 0 else
        'Q3' if row['x'] < 0 and row['y'] < 0 else
        'Q4'
    ), axis=1)

    # Get all feature columns (x, y, and any noise columns)
    feature_cols = ['x', 'y'] + \
        [col for col in df.columns if col.startswith('noise_')]

    # Initialize results dictionary
    results = {}

    # Analyze each label separately
    for label in df['label'].unique():
        label_data = df[df['label'] == label]

        # Count samples in each quadrant
        quadrant_counts = label_data['quadrant'].value_counts()

        # Calculate statistics for each quadrant
        quadrant_stats = {}
        for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
            quadrant_data = label_data[label_data['quadrant'] == quadrant]

            if not quadrant_data.empty:
                stats = {
                    'count': len(quadrant_data),
                    'features': {}
                }

                # Calculate statistics for each feature
                for feature in feature_cols:
                    stats['features'][feature] = {
                        'min': quadrant_data[feature].min(),
                        'max': quadrant_data[feature].max(),
                        'median': quadrant_data[feature].median()
                    }
            else:
                stats = {'count': 0, 'features': {}}

            quadrant_stats[quadrant] = stats

        results[label] = quadrant_stats

    return results


def print_analysis(results, filepath):
    """Print analysis results in a formatted way."""
    logger.info(f"\nAnalysis for: {filepath}")
    logger.info("=" * 80)

    for label, quadrant_data in results.items():
        logger.info(f"\nLabel {label}:")
        logger.info("-" * 40)

        total_samples = sum(qd['count'] for qd in quadrant_data.values())
        logger.info(f"Total samples: {total_samples}")

        for quadrant, stats in quadrant_data.items():
            if stats['count'] > 0:
                percentage = (stats['count'] / total_samples) * 100
                logger.info(
                    f"\n{quadrant}: {stats['count']} samples ({percentage:.1f}%)")

                for feature, feature_stats in stats['features'].items():
                    logger.info(
                        f"  {feature:8}: "
                        f"min = {feature_stats['min']:8.3f}, "
                        f"max = {feature_stats['max']:8.3f}, "
                        f"median = {feature_stats['median']:8.3f}"
                    )


def main():
    # Find all data.csv files
    data_files = glob.glob("data/raw/*_data.csv")

    if not data_files:
        logger.error("No data files found in data/raw/ directory")
        return

    for filepath in sorted(data_files):
        try:
            # Load data
            df = pd.read_csv(filepath)

            # Analyze data
            results = analyze_quadrants(df)

            # Print results
            print_analysis(results, filepath)

        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")


if __name__ == "__main__":
    main()
