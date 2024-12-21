# The script plots data from CSV files in a 'data/raw' directory. It uses
# pandas to read files, matplotlib and seaborn for plotting. It creates
# subplots for each file, plotting numerical columns against the index.
# The script sets titles, labels, and legends for each subplot, then
# displays and saves the figure. It assumes CSV format with plottable
# numerical data.


import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model


def plot_data_from_directory(data_dir):
    try:
        plt.style.use('ggplot')  # Using 'ggplot' style instead of 'seaborn'
    except:
        print("Warning: 'ggplot' style not available. Using default style.")

    sns.set_palette("deep")

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    if not files:
        print(f"No CSV files found in {data_dir}")
        return

    fig, axes = plt.subplots(len(files), 1, figsize=(
        12, 6*len(files)), squeeze=False)
    fig.suptitle('Data from raw files', fontsize=16)

    for i, file in enumerate(files):
        try:
            df = pd.read_csv(os.path.join(data_dir, file))

            # Ensure there are at least three columns
            if df.shape[1] < 3:
                print(
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
            print(f"Plot saved for {file} as 'plots/{file}_scatter.png'.")
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")


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
        print("Warning: 'ggplot' style not available. Using default style.")

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
    print(f"Final evaluation plot saved as '{save_path}'.")


def plot_train_test_with_decision_boundary(model, X_train, X_test, y_train, y_test, save_path='plots/train_test_decision_boundary.png'):
    """
    Plots training and testing data colored by class with decision boundaries on an A4 landscape sheet.
    Differentiates correctly classified and misclassified data points using different markers.

    Parameters:
    - model: Trained Keras model with a predict method.
    - X_train (np.ndarray): Training feature data.
    - X_test (np.ndarray): Testing feature data.
    - y_train (np.ndarray): Training labels.
    - y_test (np.ndarray): Testing labels.
    - save_path (str): Path to save the plot (determines format based on file extension).
    """
    # Set plot style
    try:
        plt.style.use('ggplot')
    except:
        print("Warning: 'ggplot' style not available. Using default style.")
    sns.set_palette("deep")

    # Create A4 landscape figure
    # A4 landscape size in inches
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))

    # Define titles with current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    fig.suptitle(
        f'Training and Testing Data with Decision Boundary\n{current_date}', fontsize=16)

    # Define markers for correct and misclassified points
    markers = {True: 'o', False: 'X'}  # 'o' for correct, 'X' for misclassified
    marker_labels = {True: 'Correctly Classified', False: 'Misclassified'}

    for ax, data, labels, data_type in zip(
        axes,
        [X_train, X_test],
        [y_train, y_test],
        ['Training', 'Testing']
    ):
        # Predict classes for the current dataset
        predictions = (model.predict(data) > 0.5).astype(int).reshape(-1)
        correctness = predictions == labels

        # Separate correct and misclassified points
        correct_data = data[correctness]
        correct_labels = labels[correctness]
        misclassified_data = data[~correctness]
        misclassified_labels = labels[~correctness]

        # Scatter plot for correctly classified points
        scatter_correct = ax.scatter(
            correct_data[:, 0],
            correct_data[:, 1],
            c=correct_labels,
            cmap='viridis',
            marker=markers[True],
            alpha=0.7,
            edgecolors='w',
            linewidth=0.5,
            label='Correctly Classified'
        )

        # Scatter plot for misclassified points
        scatter_misclassified = ax.scatter(
            misclassified_data[:, 0],
            misclassified_data[:, 1],
            c=misclassified_labels,
            cmap='viridis',
            marker=markers[False],
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5,
            label='Misclassified'
        )

        # Create mesh grid for decision boundary
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict classes for each point in the grid
        Z = model.predict(grid)
        Z = (Z > 0.5).astype(int).reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')

        # Set labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{data_type} Data')

    # Create a combined legend
    handles_correct, labels_correct = scatter_correct.legend_elements(
        prop="colors")
    handles_misclassified, labels_misclassified = scatter_misclassified.legend_elements(
        prop="colors")

    # Create custom legend handles for correctness
    custom_handles = [
        Line2D([0], [0], marker=markers[True], color='w', label='Correctly Classified',
               markerfacecolor='gray', markersize=10, markeredgecolor='w'),
        Line2D([0], [0], marker=markers[False], color='w', label='Misclassified',
               markerfacecolor='gray', markersize=10, markeredgecolor='k')
    ]

    # Add class legends
    class_unique = np.unique(y_train)
    class_colors = [plt.cm.plasma(
        i / float(len(class_unique)-1)) for i in class_unique]
    class_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {cls}',
                                markerfacecolor=color, markersize=10) for cls, color in zip(class_unique, class_colors)]

    # Combine all legends
    first_legend = ax.legend(handles=class_handles,
                             title="Classes", loc='upper right')
    second_legend = ax.add_artist(plt.legend(custom_handles, [h.get_label() for h in custom_handles],
                                             title="Classification", loc='lower right'))

    # Adjust layout to make room for the suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(
        f"Train and Test plots with decision boundary saved as '{save_path}'.")


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
        print(f"Model file '{model_path}' does not exist.")
        return

    # Load the Keras model
    model = load_model(model_path)
    print(f"Model loaded from '{model_path}'.")

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
