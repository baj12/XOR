# The script plots data from CSV files in a 'data/raw' directory. It uses
# pandas to read files, matplotlib and seaborn for plotting. It creates
# subplots for each file, plotting numerical columns against the index.
# The script sets titles, labels, and legends for each subplot, then
# displays and saves the figure. It assumes CSV format with plottable
# numerical data.


import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)


def plot_data_from_directory(data_dir):
    try:
        plt.style.use('ggplot')  # Using 'ggplot' style instead of 'seaborn'
    except:
        logger.error(
            "Warning: 'ggplot' style not available. Using default style.")

    sns.set_palette("deep")

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not files:
        logger.error(f"No CSV files found in {data_dir}")
        return

    fig, axes = plt.subplots(len(files), 1, figsize=(
        12, 6*len(files)), squeeze=False)
    fig.suptitle('Data from raw files', fontsize=16)

    for i, file in enumerate(files):
        try:
            df = pd.read_csv(os.path.join(data_dir, file))

            # Ensure there are at least three columns
            if df.shape[1] < 3:
                logger.error(
                    f"File {file} does not have at least three columns. Skipping.")
                continue

            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            color = df.iloc[:, 2]

            plt.figure(figsize=(12, 6))
            scatter = plt.scatter(x, y, c=color, cmap='viridis',
                                  alpha=0.7, edgecolors='w', linewidth=0.5)
            plt.title(f'Data from {file}')
            plt.xlabel(df.columns[0])
            plt.ylabel(df.columns[1])
            cbar = plt.colorbar(scatter)
            cbar.set_label(df.columns[2])
            plt.tight_layout()
            plt.savefig(f'plots/{file}_scatter.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(
                f"Plot saved for {file} as 'plots/{file}_scatter.png'.")
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")


def plot_final_evaluation(df, predictions, save_path='plots/final_evaluation.png'):
    """
    Plots the 2D data colored by actual class and overlays predicted class boundaries.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'x', 'y', and 'label' columns.
    - predictions (np.ndarray): Predicted labels for the data points.
    - save_path (str): Path to save the final evaluation plot.
    """
    try:
        plt.style.use('ggplot')
    except:
        logger.error(
            "Warning: 'ggplot' style not available. Using default style.")

    sns.set_palette("deep")

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df['x'], df['y'], c=df['label'], cmap='viridis',
                          alpha=0.7, edgecolors='w', linewidth=0.5, label='Actual')
    plt.scatter(df['x'], df['y'], c=predictions, cmap='coolwarm',
                alpha=0.3, marker='x', label='Predicted')
    plt.title('Final Evaluation: Actual vs Predicted Classes')
    plt.xlabel('x')
    plt.ylabel('y')
    handles, labels = scatter.legend_elements()
    plt.legend(*scatter.legend_elements(),
               title="Classes", loc='upper right')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Final evaluation plot saved as '{save_path}'.")


def plot_train_test_with_decision_boundary(model, X_train, X_test, y_train, y_test, df=None, config=None, save_path='plots/train_test_decision_boundary.png'):
    """
    Creates two separate plots: one for training data and one for test data with decision boundaries.
    """
    # Create mesh grid for the base features (x, y)
    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    # If we have noise dimensions, prepare noise values for prediction
    if config and config.experiment.noise_dimensions > 0:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        noise_cols = [
            f'noise_{i+1}' for i in range(config.experiment.noise_dimensions)]
        noise_means = df[noise_cols].mean().values
        noise_array = np.tile(noise_means, (len(mesh_points), 1))
        mesh_points = np.column_stack([mesh_points, noise_array])
    else:
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

    # Predict
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Training Data
    contour1 = ax1.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter1 = ax1.scatter(X_train[:, 0], X_train[:, 1],
                           c=y_train, cmap='viridis',
                           marker='o', alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Training Data with Decision Boundary')
    fig.colorbar(contour1, ax=ax1)

    # Plot Test Data
    contour2 = ax2.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1],
                           c=y_test, cmap='viridis',
                           marker='s', alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Test Data with Decision Boundary')
    fig.colorbar(contour2, ax=ax2)

    # Add overall title
    fig.suptitle('Model Decision Boundaries', fontsize=16, y=1.05)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plots
    base_path = os.path.splitext(save_path)[0]
    plt.savefig(f"{base_path}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Also create separate files for training and test plots
    # Training plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap='viridis',
                marker='o', alpha=0.6)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Training Data with Decision Boundary')
    plt.savefig(f"{base_path}_train.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Test plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X_test[:, 0], X_test[:, 1],
                c=y_test, cmap='viridis',
                marker='s', alpha=0.6)
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Test Data with Decision Boundary')
    plt.savefig(f"{base_path}_test.png", bbox_inches='tight', dpi=300)
    plt.close()


def plot_with_model_file(model_path, X_train, X_test, y_train, y_test, save_path='plots/train_test_decision_boundary.pdf'):
    """
    Loads a Keras model from a file and plots training and testing data colored by class with decision boundaries.

    Parameters:
    - model_path (str): Path to the saved Keras model file (.h5).
    - X_train (np.ndarray): Training feature data.
    - X_test (np.ndarray): Testing feature data.
    - y_train (np.ndarray): Training labels.
    - y_test (np.ndarray): Testing labels.
    - save_path (str): Path to save the PDF plot.
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' does not exist.")
        return

    # Load the Keras model
    model = load_model(model_path)
    logger.info(f"Model loaded from '{model_path}'.")

    # Plot using the loaded model
    plot_train_test_with_decision_boundary(
        model, X_train, X_test, y_train, y_test, save_path)


def main():
    data_dir = 'data/raw'
    plot_data_from_directory(data_dir)

    model_path = "results/ga_results_20211001_123456_1234.keras"
    plot_with_model_file(model_path, X_train, X_test, y_train, y_test,
                         save_path='plots/train_test_decision_boundary_loaded_model.pdf')


if __name__ == "__main__":
    main()
