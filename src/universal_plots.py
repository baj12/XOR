import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

logger = logging.getLogger(__name__)

def create_universal_classification_plots(model, X_train, X_val, y_train, y_val, save_path, 
                                        title_prefix="Classification", umap_params=None):
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
    
    return X_umap, reducer  # Return UMAP embedding and reducer for future use


def create_feature_importance_plot(model, save_path, max_features=50):
    """Create feature importance plot from first layer weights"""
    try:
        # Get first layer weights
        first_layer_weights = model.layers[0].get_weights()[0]
        feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
        
        plt.figure(figsize=(12, 6))
        
        # Plot all features
        plt.subplot(1, 2, 1)
        plt.plot(feature_importance, alpha=0.7)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance (All Features)')
        plt.grid(True, alpha=0.3)
        
        # Plot top features
        plt.subplot(1, 2, 2)
        top_features = np.argsort(feature_importance)[-max_features:]
        plt.barh(range(len(top_features)), feature_importance[top_features])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature Rank')
        plt.title(f'Top {max_features} Most Important Features')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Could not create feature importance plot: {e}")
