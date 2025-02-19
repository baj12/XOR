import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_data(df, save_path=None, plot_noise=False):
    """
    Plot the XOR data with various visualizations.

    Args:
        df: DataFrame containing x, y, label and optional noise columns
        save_path: Path to save the plots (optional)
        plot_noise: Whether to create additional plots for noise dimensions
    """
    # Set style
    sns.set_theme(style="whitegrid")

    # Create figure with multiple subplots
    if plot_noise and any(col.startswith('noise_') for col in df.columns):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot 1: Basic scatter plot
    sns.scatterplot(data=df, x='x', y='y', hue='label',
                    style='label', ax=ax1)
    ax1.set_title('XOR Data Distribution')
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    # Add quadrant lines
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Density plot
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        sns.kdeplot(data=label_data, x='x', y='y',
                    levels=5, alpha=0.5, ax=ax2,
                    label=f'Class {label}')
    ax2.set_title('Density Distribution by Class')
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)

    if plot_noise and any(col.startswith('noise_') for col in df.columns):
        # Plot 3: Noise distributions
        noise_cols = [col for col in df.columns if col.startswith('noise_')]
        for noise_col in noise_cols:
            sns.kdeplot(data=df, x=noise_col, hue='label', ax=ax3)
        ax3.set_title('Noise Distributions by Class')

        # Plot 4: Noise correlation matrix
        plot_cols = ['x', 'y'] + noise_cols
        sns.heatmap(df[plot_cols].corr(), annot=True,
                    cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Feature Correlation Matrix')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_quadrant_statistics(df, save_path=None):
    """
    Plot statistics about quadrant distribution and class separation.
    """
    sns.set_theme(style="whitegrid")

    # Define quadrants
    df['quadrant'] = df.apply(lambda row: (
        'Q1' if row['x'] >= 0 and row['y'] >= 0 else
        'Q2' if row['x'] < 0 and row['y'] >= 0 else
        'Q3' if row['x'] < 0 and row['y'] < 0 else
        'Q4'
    ), axis=1)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot 1: Quadrant distribution by class
    quadrant_counts = pd.crosstab(df['quadrant'], df['label'])
    quadrant_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Sample Distribution by Quadrant and Class')
    ax1.set_ylabel('Number of Samples')

    # Plot 2: Distance from origin distribution
    df['distance'] = np.sqrt(df['x']**2 + df['y']**2)
    sns.boxplot(data=df, x='quadrant', y='distance',
                hue='label', ax=ax2)
    ax2.set_title('Distance Distribution by Quadrant and Class')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot XOR dataset")
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data CSV file')
    parser.add_argument('--output', type=str, default=None,
                        help='Directory to save plots (optional)')
    parser.add_argument('--plot-noise', action='store_true',
                        help='Include noise distribution plots')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data)
    print(f"Loaded data shape: {df.shape}")
    print("\nColumn names:", df.columns.tolist())
    print("\nSample of data:")
    print(df.head())

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(args.data))[0]
        main_plot_path = os.path.join(
            args.output, f"{base_name}_distribution.png")
        stats_plot_path = os.path.join(
            args.output, f"{base_name}_statistics.png")
    else:
        main_plot_path = None
        stats_plot_path = None

    # Generate plots
    plot_data(df, main_plot_path, args.plot_noise)
    plot_quadrant_statistics(df, stats_plot_path)


if __name__ == "__main__":
    main()
