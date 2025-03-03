import pandas as pd
import numpy as np
from scipy import stats
import glob
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_value_ranges(df: pd.DataFrame, allowed_overflow: float = 0.01) -> Dict:
    """Check if values are within expected ranges with allowed overflow."""
    results = {}
    
    # Check x and y columns
    for col in ['x', 'y']:
        values = df[col].abs()
        within_range = (values <= 1.0)
        overflow_pct = 1 - (within_range.sum() / len(df))
        max_val = values.max()
        
        results[col] = {
            'within_range_pct': (1 - overflow_pct) * 100,
            'max_abs_value': max_val,
            'valid': overflow_pct <= allowed_overflow and max_val <= 1.1  # Allow 10% overflow
        }
    
    # Check noise columns
    noise_cols = [col for col in df.columns if col.startswith('noise_')]
    for col in noise_cols:
        values = df[col].abs()
        within_range = (values <= 1.0)
        overflow_pct = 1 - (within_range.sum() / len(df))
        max_val = values.max()
        
        results[col] = {
            'within_range_pct': (1 - overflow_pct) * 100,
            'max_abs_value': max_val,
            'valid': overflow_pct <= allowed_overflow and max_val <= 1.1
        }
    
    return results

def check_quadrant_distribution(df: pd.DataFrame) -> Dict:
    """Check if data is evenly distributed across quadrants."""
    
    # Define quadrants
    df['quadrant'] = df.apply(lambda row: (
        'Q1' if row['x'] >= 0 and row['y'] >= 0 else
        'Q2' if row['x'] < 0 and row['y'] >= 0 else
        'Q3' if row['x'] < 0 and row['y'] < 0 else
        'Q4'
    ), axis=1)
    
    # Count samples in each quadrant per label
    distribution = {}
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        counts = label_data['quadrant'].value_counts()
        expected_count = len(label_data) / 4
        
        # Chi-square test for uniform distribution
        chi2, p_value = stats.chisquare(counts)
        
        distribution[label] = {
            'counts': counts.to_dict(),
            'chi2_stat': chi2,
            'p_value': p_value,
            'is_uniform': p_value > 0.05  # Using 5% significance level
        }
    
    return distribution

def check_class_separation(df: pd.DataFrame, separation: float) -> Dict:
    """Check if class separation matches the specified value."""
    results = {}
    
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        
        # For each quadrant that should contain this label
        expected_ranges = []
        if label == 1:  # Q1 and Q3
            q1_data = label_data[
                (label_data['x'] >= 0) & (label_data['y'] >= 0)]
            q3_data = label_data[
                (label_data['x'] < 0) & (label_data['y'] < 0)]
            expected_ranges.extend([q1_data, q3_data])
        else:  # Q2 and Q4
            q2_data = label_data[
                (label_data['x'] < 0) & (label_data['y'] >= 0)]
            q4_data = label_data[
                (label_data['x'] >= 0) & (label_data['y'] < 0)]
            expected_ranges.extend([q2_data, q4_data])
        
        # Check separation in each relevant quadrant
        for i, quadrant_data in enumerate(expected_ranges):
            min_dist = min(abs(quadrant_data['x'].min()), 
                         abs(quadrant_data['x'].max()),
                         abs(quadrant_data['y'].min()),
                         abs(quadrant_data['y'].max()))
            
            expected_min = 1 - separation
            results[f'label_{label}_quadrant_{i}'] = {
                'min_distance': min_dist,
                'expected_min': expected_min,
                'valid': abs(min_dist - expected_min) < 0.1  # Allow 10% deviation
            }
    
    return results

def check_noise_randomness(df: pd.DataFrame) -> Dict:
    """Check if noise columns are random with respect to XOR problem."""
    results = {}
    noise_cols = [col for col in df.columns if col.startswith('noise_')]
    
    for noise_col in noise_cols:
        # Check correlation with x and y
        corr_x = stats.pearsonr(df['x'], df[noise_col])[0]
        corr_y = stats.pearsonr(df['y'], df[noise_col])[0]
        
        # Check if noise distribution is different for different labels
        noise_by_label = [df[df['label'] == label][noise_col] 
                         for label in df['label'].unique()]
        if len(noise_by_label) > 1:
            label_diff = stats.ks_2samp(noise_by_label[0], noise_by_label[1])
        else:
            label_diff = None
        
        results[noise_col] = {
            'correlation_x': corr_x,
            'correlation_y': corr_y,
            'label_independence_pvalue': label_diff.pvalue if label_diff else None,
            'is_random': (
                abs(corr_x) < 0.1 and 
                abs(corr_y) < 0.1 and 
                (label_diff is None or label_diff.pvalue > 0.05)
            )
        }
    
    return results

def main():
    data_files = glob.glob("data/raw/*_data.csv")
    
    if not data_files:
        logger.error("No data files found in data/raw/ directory")
        return
    
    for filepath in sorted(data_files):
        try:
            logger.info(f"\nAnalyzing: {filepath}")
            logger.info("=" * 80)
            
            df = pd.read_csv(filepath)
            
            # Check value ranges
            range_results = check_value_ranges(df)
            logger.info("\nValue Range Check:")
            for col, stats in range_results.items():
                logger.info(f"{col}: {stats['within_range_pct']:.2f}% within range "
                          f"(max abs: {stats['max_abs_value']:.3f}) "
                          f"{'✓' if stats['valid'] else '✗'}")
            
            # Check quadrant distribution
            dist_results = check_quadrant_distribution(df)
            logger.info("\nQuadrant Distribution Check:")
            for label, stats in dist_results.items():
                logger.info(f"\nLabel {label}:")
                for q, count in stats['counts'].items():
                    logger.info(f"  {q}: {count}")
                logger.info(f"Uniform distribution test p-value: {stats['p_value']:.4f} "
                          f"{'✓' if stats['is_uniform'] else '✗'}")
            
            # Check class separation
            # Try to extract separation value from filename or use default
            separation = 1.0  # Default value
            sep_results = check_class_separation(df, separation)
            logger.info("\nClass Separation Check:")
            for quad, stats in sep_results.items():
                logger.info(f"{quad}: min distance = {stats['min_distance']:.3f} "
                          f"(expected: {stats['expected_min']:.3f}) "
                          f"{'✓' if stats['valid'] else '✗'}")
            
            # Check noise randomness
            noise_results = check_noise_randomness(df)
            logger.info("\nNoise Randomness Check:")
            for noise_col, stats in noise_results.items():
                logger.info(f"\n{noise_col}:")
                logger.info(f"  Correlation with x: {stats['correlation_x']:.3f}")
                logger.info(f"  Correlation with y: {stats['correlation_y']:.3f}")
                if stats['label_independence_pvalue'] is not None:
                    logger.info(f"  Label independence p-value: "
                              f"{stats['label_independence_pvalue']:.3f}")
                logger.info(f"  Random: {'✓' if stats['is_random'] else '✗'}")
            
        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")

if __name__ == "__main__":
    main()


