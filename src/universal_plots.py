import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from scipy import stats

logger = logging.getLogger(__name__)

def create_universal_classification_plots(model, X_train, X_val, y_train, y_val, save_path,
                                        title_prefix="Classification", umap_params=None,
                                        include_pca=False, include_tsne=False):
    """
    Create universal classification plots using UMAP for dimensionality reduction.
    Works for any input dimensionality (2D XOR, high-dimensional audio, etc.)

    Args:
        model: Trained model
        X_train, X_val: Training and validation features
        y_train, y_val: Training and validation labels
        save_path: Path to save the plot
        title_prefix: Prefix for plot titles
        umap_params: Dictionary of UMAP parameters (optional)
        include_pca: If True, also create PCA plots (saved separately)
        include_tsne: If True, also create t-SNE plots (saved separately)
    """
    
    if umap_params is None:
        umap_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'n_components': 2,
            'metric': 'euclidean',
            'random_state': 42
        }
    
    # Make predictions
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    
    # Convert predictions to binary
    train_pred_binary = (train_predictions > 0.5).astype(int).flatten()
    val_pred_binary = (val_predictions > 0.5).astype(int).flatten()
    
    # Combine data for UMAP
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.hstack([y_train, y_val])
    pred_combined = np.hstack([train_pred_binary, val_pred_binary])
    
    # Apply UMAP
    logger.info(f"Applying UMAP to {X_combined.shape[1]}D data...")
    reducer = umap.UMAP(**umap_params)
    X_umap = reducer.fit_transform(X_combined)
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. UMAP with true labels
    scatter1 = axes[0, 0].scatter(X_umap[:, 0], X_umap[:, 1], c=y_combined, 
                                 cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 0].set_xlabel('UMAP Component 1')
    axes[0, 0].set_ylabel('UMAP Component 2')
    axes[0, 0].set_title(f'{title_prefix} - True Labels (UMAP)')
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 2. UMAP with predicted labels
    scatter2 = axes[0, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=pred_combined, 
                                 cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 1].set_xlabel('UMAP Component 1')
    axes[0, 1].set_ylabel('UMAP Component 2')
    axes[0, 1].set_title(f'{title_prefix} - Predicted Labels (UMAP)')
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 3. UMAP with errors highlighted
    # Create error mask
    n_train = len(X_train)
    train_errors = train_pred_binary != y_train
    val_errors = val_pred_binary != y_val
    error_mask = np.hstack([train_errors, val_errors])
    
    # Plot correct and incorrect predictions
    correct_mask = ~error_mask
    axes[0, 2].scatter(X_umap[correct_mask, 0], X_umap[correct_mask, 1], 
                      c='green', alpha=0.6, s=20, label='Correct')
    axes[0, 2].scatter(X_umap[error_mask, 0], X_umap[error_mask, 1], 
                      c='red', alpha=0.8, s=20, label='Incorrect')
    axes[0, 2].set_xlabel('UMAP Component 1')
    axes[0, 2].set_ylabel('UMAP Component 2')
    axes[0, 2].set_title(f'{title_prefix} - Prediction Errors (UMAP)')
    axes[0, 2].legend()
    
    # 4. Prediction distribution
    axes[1, 0].hist(train_predictions[y_train == 0], alpha=0.5, label='Class 0 (Train)', bins=20)
    axes[1, 0].hist(train_predictions[y_train == 1], alpha=0.5, label='Class 1 (Train)', bins=20)
    axes[1, 0].hist(val_predictions[y_val == 0], alpha=0.5, label='Class 0 (Val)', bins=20)
    axes[1, 0].hist(val_predictions[y_val == 1], alpha=0.5, label='Class 1 (Val)', bins=20)
    axes[1, 0].set_xlabel('Prediction Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()
    
    # 5. Confusion Matrix
    y_true = np.hstack([y_train, y_val])
    y_pred = np.hstack([train_pred_binary, val_pred_binary])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')
    
    # 6. Per-class accuracy
    train_acc_class0 = np.mean(train_pred_binary[y_train == 0] == 0)
    train_acc_class1 = np.mean(train_pred_binary[y_train == 1] == 1)
    val_acc_class0 = np.mean(val_pred_binary[y_val == 0] == 0)
    val_acc_class1 = np.mean(val_pred_binary[y_val == 1] == 1)
    
    categories = ['Class 0', 'Class 1']
    train_accs = [train_acc_class0, train_acc_class1]
    val_accs = [val_acc_class0, val_acc_class1]
    
    x = np.arange(len(categories))
    width = 0.35
    
    axes[1, 2].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    axes[1, 2].bar(x + width/2, val_accs, width, label='Validation', alpha=0.8)
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Per-Class Accuracy')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(categories)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    overall_acc = accuracy_score(y_true, y_pred)
    logger.info(f"Overall accuracy: {overall_acc:.4f}")
    logger.info(f"Train accuracy: {accuracy_score(y_train, train_pred_binary):.4f}")
    logger.info(f"Validation accuracy: {accuracy_score(y_val, val_pred_binary):.4f}")
    logger.info(f"UMAP reduced {X_combined.shape[1]}D data to 2D for visualization")
    logger.info(f"Universal classification plots saved to {save_path}")

    # Create PCA plots if requested
    if include_pca:
        pca_save_path = save_path.replace('.png', '_pca.png')
        _create_pca_plots(X_combined, y_combined, pred_combined, error_mask,
                         train_predictions, val_predictions, y_train, y_val,
                         train_pred_binary, val_pred_binary, pca_save_path, title_prefix)

    # Create t-SNE plots if requested
    if include_tsne:
        tsne_save_path = save_path.replace('.png', '_tsne.png')
        _create_tsne_plots(X_combined, y_combined, pred_combined, error_mask,
                          train_predictions, val_predictions, y_train, y_val,
                          train_pred_binary, val_pred_binary, tsne_save_path, title_prefix)

    return X_umap, reducer  # Return UMAP embedding and reducer for future use


def _create_pca_plots(X_combined, y_combined, pred_combined, error_mask,
                     train_predictions, val_predictions, y_train, y_val,
                     train_pred_binary, val_pred_binary, save_path, title_prefix):
    """Helper function to create PCA visualization plots"""
    logger.info(f"Creating PCA plots...")

    # Apply PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_combined)

    variance_explained = pca.explained_variance_ratio_

    # Create the plot (same structure as UMAP)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. PCA with true labels
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_combined,
                                 cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 0].set_xlabel(f'PC1 ({variance_explained[0]:.2%})')
    axes[0, 0].set_ylabel(f'PC2 ({variance_explained[1]:.2%})')
    axes[0, 0].set_title(f'{title_prefix} - True Labels (PCA)\nTotal Variance: {variance_explained.sum():.2%}')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # 2. PCA with predicted labels
    scatter2 = axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], c=pred_combined,
                                 cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 1].set_xlabel(f'PC1 ({variance_explained[0]:.2%})')
    axes[0, 1].set_ylabel(f'PC2 ({variance_explained[1]:.2%})')
    axes[0, 1].set_title(f'{title_prefix} - Predicted Labels (PCA)')
    plt.colorbar(scatter2, ax=axes[0, 1])

    # 3. PCA with errors highlighted
    correct_mask = ~error_mask
    axes[0, 2].scatter(X_pca[correct_mask, 0], X_pca[correct_mask, 1],
                      c='green', alpha=0.6, s=20, label='Correct')
    axes[0, 2].scatter(X_pca[error_mask, 0], X_pca[error_mask, 1],
                      c='red', alpha=0.8, s=20, label='Incorrect')
    axes[0, 2].set_xlabel(f'PC1 ({variance_explained[0]:.2%})')
    axes[0, 2].set_ylabel(f'PC2 ({variance_explained[1]:.2%})')
    axes[0, 2].set_title(f'{title_prefix} - Prediction Errors (PCA)')
    axes[0, 2].legend()

    # 4. Prediction distribution
    axes[1, 0].hist(train_predictions[y_train == 0], alpha=0.5, label='Class 0 (Train)', bins=20)
    axes[1, 0].hist(train_predictions[y_train == 1], alpha=0.5, label='Class 1 (Train)', bins=20)
    axes[1, 0].hist(val_predictions[y_val == 0], alpha=0.5, label='Class 0 (Val)', bins=20)
    axes[1, 0].hist(val_predictions[y_val == 1], alpha=0.5, label='Class 1 (Val)', bins=20)
    axes[1, 0].set_xlabel('Prediction Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()

    # 5. Confusion Matrix
    y_true = y_combined
    y_pred = pred_combined
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')

    # 6. Per-class accuracy
    train_acc_class0 = np.mean(train_pred_binary[y_train == 0] == 0)
    train_acc_class1 = np.mean(train_pred_binary[y_train == 1] == 1)
    val_acc_class0 = np.mean(val_pred_binary[y_val == 0] == 0)
    val_acc_class1 = np.mean(val_pred_binary[y_val == 1] == 1)

    categories = ['Class 0', 'Class 1']
    train_accs = [train_acc_class0, train_acc_class1]
    val_accs = [val_acc_class0, val_acc_class1]

    x = np.arange(len(categories))
    width = 0.35

    axes[1, 2].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    axes[1, 2].bar(x + width/2, val_accs, width, label='Validation', alpha=0.8)
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Per-Class Accuracy')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(categories)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"PCA plots saved to {save_path}")


def _create_tsne_plots(X_combined, y_combined, pred_combined, error_mask,
                      train_predictions, val_predictions, y_train, y_val,
                      train_pred_binary, val_pred_binary, save_path, title_prefix):
    """Helper function to create t-SNE visualization plots"""
    logger.info(f"Creating t-SNE plots (this may take a while)...")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_combined)//2))
    X_tsne = tsne.fit_transform(X_combined)

    # Create the plot (same structure as UMAP)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. t-SNE with true labels
    scatter1 = axes[0, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_combined,
                                 cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 0].set_xlabel('t-SNE Component 1')
    axes[0, 0].set_ylabel('t-SNE Component 2')
    axes[0, 0].set_title(f'{title_prefix} - True Labels (t-SNE)')
    plt.colorbar(scatter1, ax=axes[0, 0])

    # 2. t-SNE with predicted labels
    scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=pred_combined,
                                 cmap='coolwarm', alpha=0.6, s=20)
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    axes[0, 1].set_title(f'{title_prefix} - Predicted Labels (t-SNE)')
    plt.colorbar(scatter2, ax=axes[0, 1])

    # 3. t-SNE with errors highlighted
    correct_mask = ~error_mask
    axes[0, 2].scatter(X_tsne[correct_mask, 0], X_tsne[correct_mask, 1],
                      c='green', alpha=0.6, s=20, label='Correct')
    axes[0, 2].scatter(X_tsne[error_mask, 0], X_tsne[error_mask, 1],
                      c='red', alpha=0.8, s=20, label='Incorrect')
    axes[0, 2].set_xlabel('t-SNE Component 1')
    axes[0, 2].set_ylabel('t-SNE Component 2')
    axes[0, 2].set_title(f'{title_prefix} - Prediction Errors (t-SNE)')
    axes[0, 2].legend()

    # 4. Prediction distribution
    axes[1, 0].hist(train_predictions[y_train == 0], alpha=0.5, label='Class 0 (Train)', bins=20)
    axes[1, 0].hist(train_predictions[y_train == 1], alpha=0.5, label='Class 1 (Train)', bins=20)
    axes[1, 0].hist(val_predictions[y_val == 0], alpha=0.5, label='Class 0 (Val)', bins=20)
    axes[1, 0].hist(val_predictions[y_val == 1], alpha=0.5, label='Class 1 (Val)', bins=20)
    axes[1, 0].set_xlabel('Prediction Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].legend()

    # 5. Confusion Matrix
    y_true = y_combined
    y_pred = pred_combined
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    axes[1, 1].set_title('Confusion Matrix')

    # 6. Per-class accuracy
    train_acc_class0 = np.mean(train_pred_binary[y_train == 0] == 0)
    train_acc_class1 = np.mean(train_pred_binary[y_train == 1] == 1)
    val_acc_class0 = np.mean(val_pred_binary[y_val == 0] == 0)
    val_acc_class1 = np.mean(val_pred_binary[y_val == 1] == 1)

    categories = ['Class 0', 'Class 1']
    train_accs = [train_acc_class0, train_acc_class1]
    val_accs = [val_acc_class0, val_acc_class1]

    x = np.arange(len(categories))
    width = 0.35

    axes[1, 2].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
    axes[1, 2].bar(x + width/2, val_accs, width, label='Validation', alpha=0.8)
    axes[1, 2].set_xlabel('Class')
    axes[1, 2].set_ylabel('Accuracy')
    axes[1, 2].set_title('Per-Class Accuracy')
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(categories)
    axes[1, 2].legend()
    axes[1, 2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"t-SNE plots saved to {save_path}")


def create_statistical_analysis_plots(y_true, y_pred, y_pred_proba, save_path):
    """
    Create statistical analysis plots including confidence intervals and significance tests.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
    """
    logger.info("Creating statistical analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Statistical Analysis', fontsize=16, fontweight='bold')

    # 1. Bootstrap Confidence Intervals for Accuracy
    n_bootstrap = 1000
    bootstrap_accs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        bootstrap_acc = accuracy_score(y_true[indices], y_pred[indices])
        bootstrap_accs.append(bootstrap_acc)

    axes[0, 0].hist(bootstrap_accs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ci_lower = np.percentile(bootstrap_accs, 2.5)
    ci_upper = np.percentile(bootstrap_accs, 97.5)
    mean_acc = np.mean(bootstrap_accs)

    axes[0, 0].axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.3f}')
    axes[0, 0].axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
    axes[0, 0].axvline(ci_upper, color='orange', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Accuracy')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Bootstrap Confidence Interval for Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Class Imbalance Visualization
    class_counts = np.bincount(y_true.astype(int))
    axes[0, 1].bar(['Class 0', 'Class 1'], class_counts, alpha=0.7, color=['blue', 'red'])
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Class Distribution')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add imbalance ratio text
    imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
    axes[0, 1].text(0.5, 0.95, f'Imbalance Ratio: {imbalance_ratio:.2f}:1',
                   transform=axes[0, 1].transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Prediction Confidence Distribution by Correctness
    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    axes[1, 0].hist(y_pred_proba[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
    axes[1, 0].hist(y_pred_proba[incorrect_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
    axes[1, 0].set_xlabel('Prediction Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Prediction Confidence by Correctness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. Statistical Tests Summary
    axes[1, 1].axis('off')

    # Perform chi-square test for class distribution
    expected_counts = np.array([len(y_true)/2, len(y_true)/2])
    chi2, p_value_chi2 = stats.chisquare(class_counts, expected_counts)

    # Kolmogorov-Smirnov test for prediction distribution
    ks_stat, p_value_ks = stats.ks_2samp(y_pred_proba[y_true == 0], y_pred_proba[y_true == 1])

    # Create text summary
    summary_text = "Statistical Test Results\n" + "="*40 + "\n\n"
    summary_text += f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n"
    summary_text += f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n"

    summary_text += "Class Distribution:\n"
    summary_text += f"  Class 0: {class_counts[0]} ({class_counts[0]/len(y_true)*100:.1f}%)\n"
    summary_text += f"  Class 1: {class_counts[1]} ({class_counts[1]/len(y_true)*100:.1f}%)\n"
    summary_text += f"  χ² test p-value: {p_value_chi2:.4f}\n"
    summary_text += f"  {'Balanced' if p_value_chi2 > 0.05 else 'Imbalanced'}\n\n"

    summary_text += "Prediction Distributions:\n"
    summary_text += f"  KS statistic: {ks_stat:.4f}\n"
    summary_text += f"  KS test p-value: {p_value_ks:.4f}\n"
    summary_text += f"  {'Well-separated' if p_value_ks < 0.05 else 'Overlapping'}\n\n"

    summary_text += "Prediction Confidence:\n"
    summary_text += f"  Correct (mean): {np.mean(y_pred_proba[correct_mask]):.4f}\n"
    summary_text += f"  Incorrect (mean): {np.mean(y_pred_proba[incorrect_mask]):.4f}\n"

    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Statistical analysis plots saved to {save_path}")

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'chi2_pvalue': p_value_chi2,
        'ks_pvalue': p_value_ks
    }


def create_feature_importance_plot(model, save_path, max_features=50):
    """Create feature importance plot from first layer weights"""
    try:
        # Find first layer with weights
        first_layer_weights = None
        layer_index = 0
        
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if weights:  # Check if layer has weights
                first_layer_weights = weights[0]
                layer_index = i
                break
        
        if first_layer_weights is None:
            logger.error("No layers with weights found in model")
            return
            
        logger.info(f"Using weights from layer {layer_index}: {model.layers[layer_index].name}")
        
        feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
        
        # Create figure with 3 subplots
        plt.figure(figsize=(18, 6))
        
        # Plot 1: All features importance
        plt.subplot(1, 3, 1)
        plt.plot(feature_importance, alpha=0.7)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance (All {len(feature_importance)} Features)')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Top features
        plt.subplot(1, 3, 2)
        top_features = np.argsort(feature_importance)[-max_features:]
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Rank')
        plt.title(f'Top {max_features} Most Important Features')
        plt.gca().invert_yaxis()
        
        # Plot 3: Weight Distribution
        plt.subplot(1, 3, 3)
        all_weights = first_layer_weights.flatten()
        
        # Histogram
        plt.hist(all_weights, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add statistics
        mean_weight = np.mean(all_weights)
        std_weight = np.std(all_weights)
        median_weight = np.median(all_weights)
        
        # Add vertical lines for statistics
        plt.axvline(mean_weight, color='red', linestyle='--', label=f'Mean: {mean_weight:.4f}')
        plt.axvline(median_weight, color='green', linestyle='--', label=f'Median: {median_weight:.4f}')
        plt.axvline(mean_weight + std_weight, color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {std_weight:.4f}')
        plt.axvline(mean_weight - std_weight, color='orange', linestyle=':', alpha=0.7)
        
        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.title(f'Weight Distribution\n({len(all_weights)} weights)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        logger.info(f"Weight statistics - Mean: {mean_weight:.4f}, Std: {std_weight:.4f}, "
                   f"Min: {np.min(all_weights):.4f}, Max: {np.max(all_weights):.4f}")
        logger.info(f"Feature importance plot with weight distribution saved to {save_path}")
        
        
    except Exception as e:
        logger.error(f"Could not create feature importance plot: {e}")
        try:
            logger.error(f"Model layers: {[layer.name for layer in model.layers]}")
            logger.error(f"Layer weights: {[len(layer.get_weights()) for layer in model.layers]}")
        except:
            logger.error("Could not get model layer info")
