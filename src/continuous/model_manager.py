"""
Model Management and Rollback for Continuous Learning

Manages model versions, automatic rollback, and A/B comparison.

Key Features:
- Symlink-based current/previous/safe_fallback model management
- Automatic rollback on performance degradation
- Model comparison before deployment
- Version tracking and history

Usage:
    from continuous.model_manager import ModelManager

    manager = ModelManager(model_dir, db_path)

    # Save new model
    manager.save_model(model, metrics)

    # Auto-rollback if performance drops
    manager.check_and_rollback(new_metrics)

    # Manual rollback
    manager.rollback_to_previous()
    manager.rollback_to_safe()
"""

import logging
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple
import tensorflow as tf

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manage model versions with automatic rollback capability.

    Maintains three symlinks:
    - current_model.keras → latest deployed model
    - previous_model.keras → last known good model
    - safe_fallback.keras → baseline model from full retrain
    """

    def __init__(self,
                 model_dir: Path,
                 db_path: Optional[Path] = None,
                 performance_drop_threshold: float = 0.05):
        """
        Initialize model manager.

        Args:
            model_dir: Directory containing model files
            db_path: Optional database path for logging
            performance_drop_threshold: Rollback if accuracy drops > this
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.performance_threshold = performance_drop_threshold

        # Symlink paths
        self.current_link = self.model_dir / 'current_model.keras'
        self.previous_link = self.model_dir / 'previous_model.keras'
        self.safe_link = self.model_dir / 'safe_fallback.keras'

        # Metadata
        self.metadata_file = self.model_dir / 'model_metadata.json'

        logger.info(f"ModelManager initialized: {model_dir}")
        self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load model metadata from JSON"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'current': None,
                'previous': None,
                'safe_fallback': None,
                'history': []
            }
        return self.metadata

    def _save_metadata(self):
        """Save model metadata to JSON"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def save_model(self,
                  model: tf.keras.Model,
                  metrics: Dict,
                  model_type: str = 'incremental') -> Path:
        """
        Save model with version tracking and metadata.

        Args:
            model: Keras model to save
            metrics: Performance metrics (accuracy, loss, etc.)
            model_type: 'incremental', 'full_retrain', or 'baseline'

        Returns:
            Path to saved model file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'model_{timestamp}_{model_type}.keras'
        model_path = self.model_dir / model_filename

        # Save model
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save metrics alongside model
        metrics_path = model_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        # Update metadata
        model_entry = {
            'path': str(model_path),
            'timestamp': timestamp,
            'type': model_type,
            'metrics': metrics
        }
        self.metadata['history'].append(model_entry)
        self._save_metadata()

        return model_path

    def deploy_model(self, model_path: Path, update_previous: bool = True) -> bool:
        """
        Deploy a model by updating symlinks.

        Args:
            model_path: Path to model file to deploy
            update_previous: If True, update previous_model symlink to old current

        Returns:
            True if successful
        """
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False

            # Update previous link to current model (before switching)
            if update_previous and self.current_link.exists():
                current_target = self.current_link.resolve()
                if self.previous_link.exists():
                    self.previous_link.unlink()
                self.previous_link.symlink_to(current_target)
                logger.info(f"Updated previous_model → {current_target.name}")

            # Update current link
            if self.current_link.exists():
                self.current_link.unlink()
            self.current_link.symlink_to(model_path)
            logger.info(f"Deployed model: current_model → {model_path.name}")

            # Update metadata
            self.metadata['current'] = str(model_path)
            if update_previous:
                self.metadata['previous'] = str(model_path)
            self._save_metadata()

            return True

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}", exc_info=True)
            return False

    def set_safe_fallback(self, model_path: Path) -> bool:
        """
        Set safe fallback model (typically from full retrain).

        Args:
            model_path: Path to baseline model

        Returns:
            True if successful
        """
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                logger.error(f"Model not found: {model_path}")
                return False

            if self.safe_link.exists():
                self.safe_link.unlink()
            self.safe_link.symlink_to(model_path)

            self.metadata['safe_fallback'] = str(model_path)
            self._save_metadata()

            logger.info(f"Safe fallback set: {model_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to set safe fallback: {e}", exc_info=True)
            return False

    def rollback_to_previous(self) -> bool:
        """
        Rollback to previous model.

        Returns:
            True if successful
        """
        try:
            if not self.previous_link.exists():
                logger.error("No previous model available for rollback")
                return False

            previous_target = self.previous_link.resolve()

            # Update current to point to previous
            if self.current_link.exists():
                self.current_link.unlink()
            self.current_link.symlink_to(previous_target)

            self.metadata['current'] = str(previous_target)
            self._save_metadata()

            logger.warning(f"ROLLBACK: Reverted to previous model: {previous_target.name}")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}", exc_info=True)
            return False

    def rollback_to_safe(self) -> bool:
        """
        Rollback to safe fallback model.

        Returns:
            True if successful
        """
        try:
            if not self.safe_link.exists():
                logger.error("No safe fallback model available")
                return False

            safe_target = self.safe_link.resolve()

            # Update current to point to safe fallback
            if self.current_link.exists():
                self.current_link.unlink()
            self.current_link.symlink_to(safe_target)

            self.metadata['current'] = str(safe_target)
            self._save_metadata()

            logger.warning(f"ROLLBACK: Reverted to safe fallback: {safe_target.name}")
            return True

        except Exception as e:
            logger.error(f"Rollback to safe failed: {e}", exc_info=True)
            return False

    def check_and_rollback(self,
                          new_metrics: Dict,
                          previous_metrics: Optional[Dict] = None) -> bool:
        """
        Automatically rollback if new model underperforms.

        Args:
            new_metrics: Metrics from new model
            previous_metrics: Metrics from previous model (or load from metadata)

        Returns:
            True if rollback was triggered
        """
        try:
            # Get previous metrics from metadata if not provided
            if previous_metrics is None:
                if len(self.metadata['history']) < 2:
                    logger.info("Not enough history for comparison, skipping rollback check")
                    return False
                previous_metrics = self.metadata['history'][-2]['metrics']

            # Compare key metrics
            new_acc = new_metrics.get('test_accuracy', new_metrics.get('val_accuracy', 0))
            prev_acc = previous_metrics.get('test_accuracy', previous_metrics.get('val_accuracy', 0))

            accuracy_drop = prev_acc - new_acc

            if accuracy_drop > self.performance_threshold:
                logger.warning(
                    f"Performance drop detected: {accuracy_drop:.4f} "
                    f"(new={new_acc:.4f}, prev={prev_acc:.4f}, threshold={self.performance_threshold})"
                )

                # Trigger rollback
                success = self.rollback_to_previous()

                if success:
                    logger.warning("AUTO-ROLLBACK: Model reverted due to performance degradation")
                    return True
                else:
                    logger.error("AUTO-ROLLBACK FAILED")
                    return False

            else:
                logger.info(
                    f"Performance check passed: "
                    f"new={new_acc:.4f}, prev={prev_acc:.4f}, drop={accuracy_drop:.4f}"
                )
                return False

        except Exception as e:
            logger.error(f"Rollback check failed: {e}", exc_info=True)
            return False

    def compare_models(self,
                      model_a_path: Path,
                      model_b_path: Path,
                      test_data: Tuple) -> Dict:
        """
        Compare two models on test data.

        Args:
            model_a_path: Path to first model
            model_b_path: Path to second model
            test_data: Tuple of (X_test, y_test)

        Returns:
            Dict with comparison results
        """
        try:
            X_test, y_test = test_data

            # Load models
            model_a = tf.keras.models.load_model(model_a_path)
            model_b = tf.keras.models.load_model(model_b_path)

            # Predict
            y_pred_a = model_a.predict(X_test, verbose=0)
            y_pred_b = model_b.predict(X_test, verbose=0)

            # Compute metrics
            from sklearn.metrics import accuracy_score, roc_auc_score

            acc_a = accuracy_score(y_test, (y_pred_a > 0.5).astype(int))
            acc_b = accuracy_score(y_test, (y_pred_b > 0.5).astype(int))

            auc_a = roc_auc_score(y_test, y_pred_a)
            auc_b = roc_auc_score(y_test, y_pred_b)

            results = {
                'model_a': {
                    'path': str(model_a_path),
                    'accuracy': acc_a,
                    'roc_auc': auc_a
                },
                'model_b': {
                    'path': str(model_b_path),
                    'accuracy': acc_b,
                    'roc_auc': auc_b
                },
                'winner': 'model_a' if acc_a > acc_b else 'model_b',
                'accuracy_diff': abs(acc_a - acc_b),
                'auc_diff': abs(auc_a - auc_b)
            }

            logger.info(
                f"Model comparison: A={acc_a:.4f}, B={acc_b:.4f}, "
                f"Winner={results['winner']}"
            )

            return results

        except Exception as e:
            logger.error(f"Model comparison failed: {e}", exc_info=True)
            return {}

    def get_current_model(self) -> Optional[tf.keras.Model]:
        """
        Load and return current production model.

        Returns:
            Loaded model or None if not found
        """
        try:
            if not self.current_link.exists():
                logger.warning("No current model deployed")
                return None

            model_path = self.current_link.resolve()
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded current model: {model_path.name}")
            return model

        except Exception as e:
            logger.error(f"Failed to load current model: {e}", exc_info=True)
            return None

    def get_model_history(self, n: int = 10) -> list:
        """
        Get recent model history.

        Args:
            n: Number of recent models to return

        Returns:
            List of model entries
        """
        return self.metadata['history'][-n:]

    def cleanup_old_models(self, keep_recent: int = 10) -> int:
        """
        Delete old model files, keeping only recent ones.

        Args:
            keep_recent: Number of recent models to keep

        Returns:
            Number of models deleted
        """
        try:
            # Get all model files
            model_files = sorted(
                self.model_dir.glob('model_*.keras'),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )

            # Keep only recent ones (plus symlinked models)
            symlinked = set()
            for link in [self.current_link, self.previous_link, self.safe_link]:
                if link.exists():
                    symlinked.add(link.resolve())

            to_keep = set(model_files[:keep_recent]) | symlinked
            to_delete = set(model_files) - to_keep

            # Delete old models
            deleted_count = 0
            for model_path in to_delete:
                model_path.unlink()
                # Also delete associated metadata
                metrics_path = model_path.with_suffix('.json')
                if metrics_path.exists():
                    metrics_path.unlink()
                deleted_count += 1

            logger.info(f"Cleaned up {deleted_count} old models (kept {len(to_keep)})")
            return deleted_count

        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)
            return 0
