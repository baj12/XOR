"""
advanced_visualizations.py

Advanced visualization functions for XOR and Audio classification modes.

This module provides comprehensive analysis and visualization tools including:
- Before/after network embedding comparisons (UMAP and PCA)
- Layer-by-layer progression visualizations
- ROC and Precision-Recall curves
- Audio-specific feature analysis
- Model comparison dashboards
- Weight distribution evolution

Created for comprehensive user manuals and deep model interpretation.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, auc, average_precision_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from tensorflow import keras

logger = logging.getLogger(__name__)


def create_embedding_comparison_plots(
    model: keras.Model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_path: str,
    mode: str = 'xor'
) -> None:
    """
    Compare raw features vs. learned representations using UMAP and PCA.

    Creates 4-panel plot:
    - Top-left: Raw features (UMAP)
    - Top-right: Final hidden layer activations (UMAP)
    - Bottom-left: Raw features (PCA)
    - Bottom-right: Final hidden layer activations (PCA)

    Args:
        model: Trained Keras model
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_path: Path to save output plot
        mode: 'xor' or 'audio' for title customization
    """
    logger.info(f"Creating embedding comparison plots for {mode} mode...")

    # Combine train and test for consistent projection
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])
    n_train = len(X_train)

    # Extract final hidden layer activations
    logger.info("Extracting final hidden layer activations...")

    # For Sequential models, we need to extract features by creating a new model
    # that outputs from a specific layer
    try:
        # Try the standard approach for functional models
        _ = model.predict(X_combined[:1], verbose=0)
        embedding_model = keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output
        )
    except (AttributeError, ValueError):
        # For Sequential models without explicit input, recreate with same weights
        # Use a functional approach
        input_layer = keras.layers.Input(shape=(X_combined.shape[1],))
        x = input_layer

        # Apply all layers except the last one
        for layer in model.layers[:-1]:
            x = layer(x)

        embedding_model = keras.Model(inputs=input_layer, outputs=x)

    embeddings = embedding_model.predict(X_combined, verbose=0)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Before/After Network: Feature Space Comparison ({mode.upper()} Mode)',
                 fontsize=16, fontweight='bold', y=0.995)

    # ===== UMAP Projections =====
    logger.info("Computing UMAP projections...")

    # Raw features UMAP
    umap_raw = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    raw_2d = umap_raw.fit_transform(X_combined)

    # Plot raw features UMAP
    scatter1 = axes[0, 0].scatter(
        raw_2d[:, 0], raw_2d[:, 1],
        c=y_combined, cmap='coolwarm',
        alpha=0.6, s=30, edgecolors='k', linewidth=0.3
    )
    axes[0, 0].set_title('Raw Input Features (UMAP)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP Component 1', fontsize=11)
    axes[0, 0].set_ylabel('UMAP Component 2', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0], label='Class')

    # Add train/test boundary marker
    axes[0, 0].axhline(y=raw_2d[n_train, 1], color='gray', linestyle='--',
                       alpha=0.5, linewidth=1.5, label='Train/Test Split')
    axes[0, 0].legend(loc='upper right', fontsize=9)

    # Learned embeddings UMAP
    umap_learned = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )
    learned_2d = umap_learned.fit_transform(embeddings)

    # Plot learned embeddings UMAP
    scatter2 = axes[0, 1].scatter(
        learned_2d[:, 0], learned_2d[:, 1],
        c=y_combined, cmap='coolwarm',
        alpha=0.6, s=30, edgecolors='k', linewidth=0.3
    )
    axes[0, 1].set_title(f'Learned Representations (UMAP)\n[Final Hidden Layer: {embeddings.shape[1]} dims]',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('UMAP Component 1', fontsize=11)
    axes[0, 1].set_ylabel('UMAP Component 2', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1], label='Class')
    axes[0, 1].axhline(y=learned_2d[n_train, 1], color='gray', linestyle='--',
                       alpha=0.5, linewidth=1.5)

    # ===== PCA Projections =====
    logger.info("Computing PCA projections...")

    # Raw features PCA
    pca_raw = PCA(n_components=2, random_state=42)
    raw_pca_2d = pca_raw.fit_transform(X_combined)
    variance_raw = pca_raw.explained_variance_ratio_

    # Plot raw features PCA
    scatter3 = axes[1, 0].scatter(
        raw_pca_2d[:, 0], raw_pca_2d[:, 1],
        c=y_combined, cmap='coolwarm',
        alpha=0.6, s=30, edgecolors='k', linewidth=0.3
    )
    axes[1, 0].set_title(f'Raw Input Features (PCA)\nExplained Variance: {variance_raw[0]:.2%} + {variance_raw[1]:.2%} = {variance_raw.sum():.2%}',
                         fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel(f'PC1 ({variance_raw[0]:.2%})', fontsize=11)
    axes[1, 0].set_ylabel(f'PC2 ({variance_raw[1]:.2%})', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=axes[1, 0], label='Class')
    axes[1, 0].axhline(y=raw_pca_2d[n_train, 1], color='gray', linestyle='--',
                       alpha=0.5, linewidth=1.5)

    # Learned embeddings PCA
    pca_learned = PCA(n_components=2, random_state=42)
    learned_pca_2d = pca_learned.fit_transform(embeddings)
    variance_learned = pca_learned.explained_variance_ratio_

    # Plot learned embeddings PCA
    scatter4 = axes[1, 1].scatter(
        learned_pca_2d[:, 0], learned_pca_2d[:, 1],
        c=y_combined, cmap='coolwarm',
        alpha=0.6, s=30, edgecolors='k', linewidth=0.3
    )
    axes[1, 1].set_title(f'Learned Representations (PCA)\nExplained Variance: {variance_learned[0]:.2%} + {variance_learned[1]:.2%} = {variance_learned.sum():.2%}',
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel(f'PC1 ({variance_learned[0]:.2%})', fontsize=11)
    axes[1, 1].set_ylabel(f'PC2 ({variance_learned[1]:.2%})', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=axes[1, 1], label='Class')
    axes[1, 1].axhline(y=learned_pca_2d[n_train, 1], color='gray', linestyle='--',
                       alpha=0.5, linewidth=1.5)

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Embedding comparison plots saved to: {output_path}")

    # Log separation metrics
    logger.info("Separation analysis:")
    logger.info(f"  Raw features (UMAP) - variance: {np.var(raw_2d, axis=0).mean():.4f}")
    logger.info(f"  Learned (UMAP) - variance: {np.var(learned_2d, axis=0).mean():.4f}")
    logger.info(f"  Raw features (PCA) - total variance explained: {variance_raw.sum():.2%}")
    logger.info(f"  Learned (PCA) - total variance explained: {variance_learned.sum():.2%}")


def create_layer_progression_plots(
    model: keras.Model,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    output_path: str,
    max_samples: int = 2000
) -> None:
    """
    Show transformation through each layer using UMAP projections.

    Creates N-panel plot where N = number of hidden layers + 1.
    Each panel shows UMAP projection of that layer's activations.

    Args:
        model: Trained Keras model
        X_sample: Sample data (will be limited to max_samples)
        y_sample: Sample labels
        output_path: Path to save output plot
        max_samples: Maximum number of samples to visualize (for performance)
    """
    logger.info("Creating layer-by-layer progression plots...")

    # Limit sample size for performance
    if len(X_sample) > max_samples:
        indices = np.random.choice(len(X_sample), max_samples, replace=False)
        X_sample = X_sample[indices]
        y_sample = y_sample[indices]

    # Get all hidden layers (exclude input and output)
    hidden_layers = [layer for layer in model.layers if 'dense' in layer.name.lower() and layer != model.layers[-1]]
    n_layers = len(hidden_layers) + 1  # +1 for input

    # Calculate grid dimensions
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    fig.suptitle('Layer-by-Layer Feature Transformation (UMAP Projections)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot input layer
    logger.info("Processing input layer...")
    umap_input = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    input_2d = umap_input.fit_transform(X_sample)

    scatter = axes[0].scatter(
        input_2d[:, 0], input_2d[:, 1],
        c=y_sample, cmap='coolwarm',
        alpha=0.7, s=20, edgecolors='k', linewidth=0.2
    )
    axes[0].set_title(f'Input Layer\n[{X_sample.shape[1]} dimensions]',
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Class')

    # Plot each hidden layer
    for idx, layer in enumerate(hidden_layers):
        logger.info(f"Processing layer {idx+1}/{len(hidden_layers)}: {layer.name}...")

        # Create model up to this layer
        # Handle Sequential models that may not have .input attribute
        try:
            layer_model = keras.Model(inputs=model.input, outputs=layer.output)
        except (AttributeError, ValueError):
            # For Sequential models, create functional model
            input_layer = keras.layers.Input(shape=(X_sample.shape[1],))
            x = input_layer
            for l in model.layers[:model.layers.index(layer)+1]:
                x = l(x)
            layer_model = keras.Model(inputs=input_layer, outputs=x)

        activations = layer_model.predict(X_sample, verbose=0)

        # UMAP projection
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42+idx)
        layer_2d = umap_reducer.fit_transform(activations)

        # Plot
        scatter = axes[idx+1].scatter(
            layer_2d[:, 0], layer_2d[:, 1],
            c=y_sample, cmap='coolwarm',
            alpha=0.7, s=20, edgecolors='k', linewidth=0.2
        )

        layer_info = layer.name.replace('dense', 'Layer')
        axes[idx+1].set_title(f'{layer_info}\n[{activations.shape[1]} neurons]',
                             fontsize=12, fontweight='bold')
        axes[idx+1].set_xlabel('UMAP-1')
        axes[idx+1].set_ylabel('UMAP-2')
        axes[idx+1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx+1], label='Class')

        # Calculate and display separation metric
        class_0_mask = y_sample == 0
        class_1_mask = y_sample == 1
        if np.any(class_0_mask) and np.any(class_1_mask):
            center_0 = np.mean(layer_2d[class_0_mask], axis=0)
            center_1 = np.mean(layer_2d[class_1_mask], axis=0)
            separation = np.linalg.norm(center_1 - center_0)
            axes[idx+1].text(0.05, 0.95, f'Sep: {separation:.2f}',
                            transform=axes[idx+1].transAxes,
                            fontsize=9, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Layer progression plots saved to: {output_path}")


def create_classification_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Creates comprehensive classification performance curves.

    Creates 3-panel plot:
    - ROC curve with AUC score
    - Precision-Recall curve with AP score
    - Threshold analysis (precision, recall, F1 vs threshold)

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save output plot
        class_names: Optional class names for labeling

    Returns:
        Dictionary of metrics (AUC, AP, best_threshold, best_f1)
    """
    logger.info("Creating classification curves...")

    if class_names is None:
        class_names = ['Class 0', 'Class 1']

    # Flatten if needed
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba.flatten()

    # Calculate ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Calculate Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)

    # Find optimal threshold using F1 score
    f1_scores = []
    for threshold in pr_thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        f1_scores.append(f1)

    best_idx = np.argmax(f1_scores)
    best_threshold = pr_thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Classification Performance Analysis', fontsize=16, fontweight='bold')

    # ===== ROC Curve =====
    axes[0].plot(fpr, tpr, color='darkorange', lw=2.5,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.500)')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Add performance markers
    axes[0].fill_between(fpr, tpr, alpha=0.2, color='darkorange')

    # ===== Precision-Recall Curve =====
    axes[1].plot(recall, precision, color='blue', lw=2.5,
                label=f'PR curve (AP = {avg_precision:.3f})')
    axes[1].axhline(y=y_true.mean(), color='red', linestyle='--', lw=2,
                   label=f'No Skill (AP = {y_true.mean():.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].legend(loc="lower left", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(recall, precision, alpha=0.2, color='blue')

    # ===== Threshold Analysis =====
    # Calculate precision, recall, F1 for range of thresholds
    thresholds_range = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []

    for thresh in thresholds_range:
        y_pred = (y_pred_proba >= thresh).astype(int)
        if len(np.unique(y_pred)) > 1:  # Avoid division by zero
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1s.append(f1_score(y_true, y_pred, zero_division=0))
        else:
            precisions.append(0)
            recalls.append(0)
            f1s.append(0)

    axes[2].plot(thresholds_range, precisions, 'b-', lw=2, label='Precision', alpha=0.7)
    axes[2].plot(thresholds_range, recalls, 'g-', lw=2, label='Recall', alpha=0.7)
    axes[2].plot(thresholds_range, f1s, 'r-', lw=2.5, label='F1 Score')

    # Mark optimal threshold
    axes[2].axvline(x=best_threshold, color='orange', linestyle='--', lw=2,
                   label=f'Optimal (F1={best_f1:.3f} @ {best_threshold:.3f})')
    axes[2].set_xlabel('Decision Threshold', fontsize=12)
    axes[2].set_ylabel('Score', fontsize=12)
    axes[2].set_title('Threshold Analysis', fontsize=14, fontweight='bold')
    axes[2].legend(loc="best", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Prepare metrics
    metrics = {
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1)
    }

    logger.info(f"Classification curves saved to: {output_path}")
    logger.info(f"Metrics: AUC={roc_auc:.3f}, AP={avg_precision:.3f}, Best F1={best_f1:.3f} @ threshold={best_threshold:.3f}")

    return metrics


def create_audio_feature_analysis(
    df_features,  # Can be DataFrame or ndarray
    feature_names: List[str],
    labels: np.ndarray,
    output_path: str,
    n_mfcc: int = 13,
    n_time_frames: int = 40
) -> None:
    """
    Audio-specific feature visualization.

    Creates multi-panel visualization:
    - MFCC coefficient heatmaps (both classes)
    - Spectral feature distributions per class
    - Feature importance ranking

    Args:
        df_features: DataFrame or ndarray with extracted features
        feature_names: List of feature names
        labels: Class labels
        output_path: Path to save output plot
        n_mfcc: Number of MFCC coefficients
        n_time_frames: Number of time frames per segment
    """
    logger.info("Creating audio feature analysis plots...")

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('Audio Feature Analysis', fontsize=16, fontweight='bold')

    # Convert to numpy array if needed
    if isinstance(df_features, pd.DataFrame):
        features_array = df_features.values
    else:
        features_array = df_features

    # Extract MFCC features (assuming they're first in the feature vector)
    mfcc_features = features_array[:, :n_mfcc*n_time_frames]

    # ===== MFCC Heatmaps =====
    # Class 0
    ax1 = fig.add_subplot(gs[0, 0])
    class_0_mask = labels == 0
    if np.any(class_0_mask):
        mfcc_class_0 = mfcc_features[class_0_mask].mean(axis=0).reshape(n_mfcc, n_time_frames)
        im1 = ax1.imshow(mfcc_class_0, aspect='auto', cmap='viridis', origin='lower')
        ax1.set_title('Average MFCC - Class 0', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Time Frames')
        ax1.set_ylabel('MFCC Coefficient')
        plt.colorbar(im1, ax=ax1, label='Coefficient Value')

    # Class 1
    ax2 = fig.add_subplot(gs[0, 1])
    class_1_mask = labels == 1
    if np.any(class_1_mask):
        mfcc_class_1 = mfcc_features[class_1_mask].mean(axis=0).reshape(n_mfcc, n_time_frames)
        im2 = ax2.imshow(mfcc_class_1, aspect='auto', cmap='viridis', origin='lower')
        ax2.set_title('Average MFCC - Class 1', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Time Frames')
        ax2.set_ylabel('MFCC Coefficient')
        plt.colorbar(im2, ax=ax2, label='Coefficient Value')

    # ===== Spectral Feature Distributions =====
    ax3 = fig.add_subplot(gs[1, :])

    # Identify spectral features (assuming naming convention)
    spectral_indices = [i for i, name in enumerate(feature_names)
                       if any(keyword in name.lower() for keyword in
                             ['spectral', 'centroid', 'rolloff', 'bandwidth', 'zero_crossing'])]

    if len(spectral_indices) > 0:
        # Select first few spectral features for visualization
        spectral_subset = spectral_indices[:min(4, len(spectral_indices))]

        # Prepare data for box plots
        plot_data = []
        plot_labels = []
        plot_hues = []

        for idx in spectral_subset:
            # Class 0
            plot_data.extend(features_array[class_0_mask, idx])
            plot_labels.extend([feature_names[idx]] * np.sum(class_0_mask))
            plot_hues.extend(['Class 0'] * np.sum(class_0_mask))

            # Class 1
            plot_data.extend(features_array[class_1_mask, idx])
            plot_labels.extend([feature_names[idx]] * np.sum(class_1_mask))
            plot_hues.extend(['Class 1'] * np.sum(class_1_mask))

        plot_df = pd.DataFrame({
            'Value': plot_data,
            'Feature': plot_labels,
            'Class': plot_hues
        })

        sns.boxplot(data=plot_df, x='Feature', y='Value', hue='Class', ax=ax3, palette='Set2')
        ax3.set_title('Spectral Feature Distributions by Class', fontsize=13, fontweight='bold')
        ax3.set_xlabel('Feature Type', fontsize=11)
        ax3.set_ylabel('Normalized Value', fontsize=11)
        ax3.tick_params(axis='x', rotation=15)
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3, axis='y')

    # ===== Feature Variance Analysis =====
    ax4 = fig.add_subplot(gs[2, :])

    # Calculate per-class variance for all features
    variances_class_0 = np.var(features_array[class_0_mask], axis=0)
    variances_class_1 = np.var(features_array[class_1_mask], axis=0)

    # Plot top 50 features by total variance
    total_variance = variances_class_0 + variances_class_1
    top_indices = np.argsort(total_variance)[-50:][::-1]

    x_pos = np.arange(len(top_indices))
    width = 0.35

    ax4.bar(x_pos - width/2, variances_class_0[top_indices], width,
           label='Class 0', alpha=0.8, color='skyblue')
    ax4.bar(x_pos + width/2, variances_class_1[top_indices], width,
           label='Class 1', alpha=0.8, color='lightcoral')

    ax4.set_title('Top 50 Features by Variance', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Feature Index (sorted by total variance)', fontsize=11)
    ax4.set_ylabel('Variance', fontsize=11)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, axis='y')

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Audio feature analysis saved to: {output_path}")


def create_model_comparison_dashboard(
    models_dict: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_path: str
) -> None:
    """
    Compare multiple models side-by-side.

    Creates 4-panel visualization:
    - Accuracy comparison bar chart
    - ROC curves overlaid
    - Confusion matrices (side by side)
    - Training time comparison

    Args:
        models_dict: Dictionary of {model_name: model} or {model_name: {'model': model, 'history': history}}
        X_test: Test features
        y_test: Test labels
        output_path: Path to save output plot
    """
    logger.info(f"Creating model comparison dashboard for {len(models_dict)} models...")

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('Model Comparison Dashboard', fontsize=16, fontweight='bold')

    # Collect metrics
    accuracies = {}
    roc_data = {}
    conf_matrices = {}
    train_times = {}

    for name, model_info in models_dict.items():
        # Handle both dict and direct model
        if isinstance(model_info, dict):
            model = model_info['model']
            history = model_info.get('history', None)
        else:
            model = model_info
            history = None

        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        # Accuracy
        accuracies[name] = accuracy_score(y_test, y_pred)

        # ROC data
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}

        # Confusion matrix
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

        # Training time (if available in history)
        if history and hasattr(history, 'history'):
            # Estimate from number of epochs
            train_times[name] = len(history.history.get('loss', []))
        else:
            train_times[name] = 0

    # ===== Accuracy Comparison =====
    ax1 = fig.add_subplot(gs[0, 0])
    model_names = list(accuracies.keys())
    acc_values = list(accuracies.values())
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))

    bars = ax1.bar(model_names, acc_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, acc in zip(bars, acc_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    # ===== ROC Curves Overlaid =====
    ax2 = fig.add_subplot(gs[0, 1])

    for name, data in roc_data.items():
        ax2.plot(data['fpr'], data['tpr'], lw=2,
                label=f"{name} (AUC={data['auc']:.3f})")

    ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.500)')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curves Comparison', fontsize=13, fontweight='bold')
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ===== Confusion Matrices =====
    ax3 = fig.add_subplot(gs[1, :])

    n_models = len(conf_matrices)
    for idx, (name, cm) in enumerate(conf_matrices.items()):
        ax_sub = plt.subplot(1, n_models, idx+1)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   square=True, ax=ax_sub)
        ax_sub.set_title(f'{name}', fontsize=11, fontweight='bold')
        ax_sub.set_ylabel('True Label' if idx == 0 else '')
        ax_sub.set_xlabel('Predicted Label')

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Model comparison dashboard saved to: {output_path}")

    # Log summary
    for name in model_names:
        logger.info(f"{name}: Accuracy={accuracies[name]:.3f}, AUC={roc_data[name]['auc']:.3f}")


def create_weight_evolution_plots(
    history_checkpoints: List[Dict],
    output_path: str,
    layer_idx: int = 0
) -> None:
    """
    Visualize how weights evolve during training.

    Creates multi-panel visualization showing:
    - Weight distribution histograms at different epochs
    - Weight magnitude evolution
    - Gradient flow indicators

    Args:
        history_checkpoints: List of dicts with 'epoch' and 'weights' keys
        output_path: Path to save output plot
        layer_idx: Index of weight tensor to visualize (default: 0, first layer)
    """
    logger.info(f"Creating weight evolution plots for {len(history_checkpoints)} checkpoints...")

    if len(history_checkpoints) == 0:
        logger.warning("No checkpoints provided.")
        return

    # Extract weight data at each checkpoint
    epochs = []
    weight_means = []
    weight_stds = []
    weight_distributions = []

    for checkpoint in history_checkpoints:
        epoch = checkpoint['epoch']
        weights = checkpoint['weights']

        if layer_idx >= len(weights):
            logger.warning(f"Layer index {layer_idx} out of range. Using layer 0.")
            layer_idx = 0

        weight_matrix = weights[layer_idx]

        epochs.append(epoch)
        weight_means.append(np.mean(weight_matrix))
        weight_stds.append(np.std(weight_matrix))
        weight_distributions.append(weight_matrix.flatten())

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Weight Evolution Over Training (Layer {layer_idx})',
                 fontsize=16, fontweight='bold')

    # ===== Weight Mean Evolution =====
    axes[0, 0].plot(epochs, weight_means, marker='o', linewidth=2, markersize=6, color='steelblue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', lw=1, alpha=0.5)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Mean Weight', fontsize=11)
    axes[0, 0].set_title('Weight Mean Evolution', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # ===== Weight Std Evolution =====
    axes[0, 1].plot(epochs, weight_stds, marker='s', linewidth=2, markersize=6, color='darkorange')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Weight Std Dev', fontsize=11)
    axes[0, 1].set_title('Weight Std Dev Evolution', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # ===== Weight Distribution Histograms =====
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))

    for i, (epoch, dist) in enumerate(zip(epochs, weight_distributions)):
        axes[1, 0].hist(dist, bins=50, alpha=0.4, color=colors[i],
                       label=f'Epoch {epoch}', edgecolor='none')

    axes[1, 0].axvline(x=0, color='red', linestyle='--', lw=2, label='Zero')
    axes[1, 0].set_xlabel('Weight Value', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('Weight Distribution Over Time', fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # ===== Weight Matrix Heatmap (Final Checkpoint) =====
    final_weights = history_checkpoints[-1]['weights'][layer_idx]

    # Limit size for visualization
    max_display = 50
    if final_weights.ndim == 2:
        if final_weights.shape[0] > max_display or final_weights.shape[1] > max_display:
            # Sample subset
            row_indices = np.linspace(0, final_weights.shape[0]-1, min(max_display, final_weights.shape[0]), dtype=int)
            col_indices = np.linspace(0, final_weights.shape[1]-1, min(max_display, final_weights.shape[1]), dtype=int)
            display_matrix = final_weights[np.ix_(row_indices, col_indices)]
            title_suffix = f' (Sampled)'
        else:
            display_matrix = final_weights
            title_suffix = ''

        im = axes[1, 1].imshow(display_matrix, aspect='auto', cmap='RdBu_r',
                              vmin=-np.abs(display_matrix).max(),
                              vmax=np.abs(display_matrix).max())
        axes[1, 1].set_title(f'Final Weight Matrix{title_suffix}', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Output Dimension', fontsize=11)
        axes[1, 1].set_ylabel('Input Dimension', fontsize=11)
        plt.colorbar(im, ax=axes[1, 1], label='Weight Value')
    else:
        # 1D weight vector
        axes[1, 1].hist(final_weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[1, 1].set_xlabel('Weight Value', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Final Weight Distribution', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Weight distribution plots saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Advanced visualizations module loaded successfully.")
    logger.info("Available functions:")
    logger.info("  - create_embedding_comparison_plots()")
    logger.info("  - create_layer_progression_plots()")
    logger.info("  - create_classification_curves()")
    logger.info("  - create_audio_feature_analysis()")
    logger.info("  - create_model_comparison_dashboard()")
    logger.info("  - create_weight_evolution_plots()")
