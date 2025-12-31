"""
Visualization Generation for Continuous Learning

Creates time-series visualizations to track model performance and data drift:
- Temporal UMAP plots (data colored by collection time)
- ROC curves overlaid across weeks
- Feature drift heatmaps
- Performance trend dashboards

Usage:
    from continuous.visualization import ContinuousVisualizer

    viz = ContinuousVisualizer(db_path, output_dir)
    viz.generate_temporal_umap(weeks=12)
    viz.generate_roc_progression(weeks=12)
    viz.generate_drift_heatmap(weeks=12)
    viz.generate_performance_dashboard(weeks=12)
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class ContinuousVisualizer:
    """
    Generate time-series visualizations for continuous learning.

    Tracks model performance and data characteristics over time.
    """

    def __init__(self, db_path: Path, output_dir: Path):
        """
        Initialize visualizer.

        Args:
            db_path: Path to feature database
            output_dir: Directory for saving visualizations
        """
        from .feature_database import FeatureDatabase

        self.db = FeatureDatabase(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ContinuousVisualizer initialized: {output_dir}")

    def generate_temporal_umap(self,
                              weeks: int = 12,
                              n_samples_per_week: int = 500) -> Optional[Path]:
        """
        Generate UMAP projection colored by data collection time.

        Shows if data distribution drifts over time. Points cluster by week
        indicate drift; well-mixed points indicate stability.

        Args:
            weeks: Number of weeks to visualize
            n_samples_per_week: Max samples per week (for speed)

        Returns:
            Path to saved plot or None if failed
        """
        if not UMAP_AVAILABLE:
            logger.error("UMAP not available. Cannot generate temporal UMAP.")
            return None

        try:
            logger.info(f"Generating temporal UMAP for last {weeks} weeks...")

            # Load data by week
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)

            X_all = []
            week_labels = []
            timestamps = []

            for week_offset in range(weeks):
                week_start = start_date + timedelta(weeks=week_offset)
                week_end = week_start + timedelta(weeks=1)

                X_week, y_week, ts_week = self.db.load_features_by_date_range(
                    start_date=week_start,
                    end_date=week_end
                )

                if len(X_week) == 0:
                    continue

                # Sample if too many points
                if len(X_week) > n_samples_per_week:
                    idx = np.random.choice(len(X_week), n_samples_per_week, replace=False)
                    X_week = X_week[idx]
                    y_week = y_week[idx]
                    ts_week = [ts_week[i] for i in idx]

                X_all.append(X_week)
                week_labels.extend([week_offset] * len(X_week))
                timestamps.extend(ts_week)

            if len(X_all) == 0:
                logger.warning("No data found for temporal UMAP")
                return None

            X_all = np.vstack(X_all)
            week_labels = np.array(week_labels)

            logger.info(f"Loaded {len(X_all)} samples from {weeks} weeks")

            # UMAP projection
            reducer = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=2,
                random_state=42,
                verbose=False
            )
            X_umap = reducer.fit_transform(X_all)

            # Plot
            fig, ax = plt.subplots(figsize=(12, 10))

            scatter = ax.scatter(
                X_umap[:, 0],
                X_umap[:, 1],
                c=week_labels,
                cmap='viridis',
                alpha=0.6,
                s=20,
                edgecolors='none'
            )

            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Weeks Ago', rotation=270, labelpad=20, fontsize=12)

            ax.set_xlabel('UMAP 1', fontsize=12)
            ax.set_ylabel('UMAP 2', fontsize=12)
            ax.set_title(
                f'Data Distribution Over Time (UMAP Projection)\n'
                f'Last {weeks} weeks, {len(X_all)} samples',
                fontsize=14,
                fontweight='bold'
            )

            # Add interpretation note
            note = (
                "Well-mixed colors = stable data distribution\n"
                "Clustered by color = data drift detected"
            )
            ax.text(
                0.02, 0.98, note,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save
            output_path = self.output_dir / f'temporal_umap_{weeks}weeks.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Temporal UMAP saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate temporal UMAP: {e}", exc_info=True)
            return None

    def generate_roc_progression(self,
                                weeks: int = 12,
                                model_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Generate ROC curves overlaid across multiple weeks.

        Shows how model performance evolves over time.

        Args:
            weeks: Number of weeks to visualize
            model_dir: Directory containing saved models

        Returns:
            Path to saved plot or None if failed
        """
        try:
            logger.info(f"Generating ROC progression for last {weeks} weeks...")

            from sklearn.metrics import roc_curve, auc
            import tensorflow as tf

            # Get training runs from last N weeks
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)

            training_runs = self.db.get_training_runs_by_date_range(
                start_date=start_date,
                end_date=end_date
            )

            if len(training_runs) == 0:
                logger.warning("No training runs found for ROC progression")
                return None

            # Load permanent test set if available
            test_set_path = Path('data/continuous/permanent_test_set.npz')
            if not test_set_path.exists():
                logger.warning("Permanent test set not found")
                return None

            test_data = np.load(test_set_path)
            X_test, y_test = test_data['X'], test_data['y']

            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot ROC curve for each training run
            colors = plt.cm.viridis(np.linspace(0, 1, len(training_runs)))

            for i, run in enumerate(training_runs):
                model_path = run['model_path']
                if not Path(model_path).exists():
                    continue

                # Load model and predict
                model = tf.keras.models.load_model(model_path)
                y_pred_proba = model.predict(X_test, verbose=0).flatten()

                # Compute ROC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                # Plot
                run_date = datetime.fromisoformat(run['run_timestamp'])
                label = f"{run_date.strftime('%Y-%m-%d')} (AUC={roc_auc:.3f})"

                ax.plot(fpr, tpr, color=colors[i], lw=2, label=label, alpha=0.7)

            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3, label='Random')

            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(
                f'ROC Curve Progression Over Time\n'
                f'Last {weeks} weeks, {len(training_runs)} models',
                fontsize=14,
                fontweight='bold'
            )

            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            plt.tight_layout()

            # Save
            output_path = self.output_dir / f'roc_progression_{weeks}weeks.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"ROC progression saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate ROC progression: {e}", exc_info=True)
            return None

    def generate_drift_heatmap(self, weeks: int = 12) -> Optional[Path]:
        """
        Generate heatmap showing per-feature drift over time.

        Uses KL divergence to measure how each feature's distribution
        changes week-to-week.

        Args:
            weeks: Number of weeks to visualize

        Returns:
            Path to saved plot or None if failed
        """
        try:
            logger.info(f"Generating drift heatmap for last {weeks} weeks...")

            from scipy.stats import entropy

            # Load data by week
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)

            drift_matrix = []
            week_dates = []

            # Baseline: first week
            baseline_week_start = start_date
            baseline_week_end = baseline_week_start + timedelta(weeks=1)
            X_baseline, _, _ = self.db.load_features_by_date_range(
                start_date=baseline_week_start,
                end_date=baseline_week_end
            )

            if len(X_baseline) == 0:
                logger.warning("No baseline data found")
                return None

            # Compute drift for each subsequent week
            for week_offset in range(1, weeks):
                week_start = start_date + timedelta(weeks=week_offset)
                week_end = week_start + timedelta(weeks=1)

                X_week, _, _ = self.db.load_features_by_date_range(
                    start_date=week_start,
                    end_date=week_end
                )

                if len(X_week) == 0:
                    continue

                # Compute KL divergence per feature
                kl_divs = []
                for feat_idx in range(min(X_baseline.shape[1], 50)):  # Limit to first 50 features
                    # Bin distributions
                    hist_baseline, bins = np.histogram(X_baseline[:, feat_idx], bins=30, density=True)
                    hist_week, _ = np.histogram(X_week[:, feat_idx], bins=bins, density=True)

                    # Add epsilon to avoid log(0)
                    hist_baseline += 1e-10
                    hist_week += 1e-10

                    # KL divergence
                    kl_div = entropy(hist_baseline, hist_week)
                    kl_divs.append(kl_div)

                drift_matrix.append(kl_divs)
                week_dates.append(week_start.strftime('%Y-%m-%d'))

            if len(drift_matrix) == 0:
                logger.warning("No drift data computed")
                return None

            drift_matrix = np.array(drift_matrix)

            # Plot heatmap
            fig, ax = plt.subplots(figsize=(14, 8))

            im = ax.imshow(
                drift_matrix.T,
                aspect='auto',
                cmap='YlOrRd',
                interpolation='nearest'
            )

            # Labels
            ax.set_xticks(np.arange(len(week_dates)))
            ax.set_xticklabels(week_dates, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Feature Index', fontsize=12)
            ax.set_xlabel('Week', fontsize=12)
            ax.set_title(
                f'Per-Feature Drift Over Time (KL Divergence)\n'
                f'Baseline: {week_dates[0] if week_dates else "N/A"}',
                fontsize=14,
                fontweight='bold'
            )

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('KL Divergence', rotation=270, labelpad=20, fontsize=12)

            # Add interpretation note
            note = (
                "Darker colors = higher drift\n"
                "Monitor features with sustained high drift"
            )
            ax.text(
                0.02, 0.98, note,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.tight_layout()

            # Save
            output_path = self.output_dir / f'drift_heatmap_{weeks}weeks.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Drift heatmap saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate drift heatmap: {e}", exc_info=True)
            return None

    def generate_performance_dashboard(self, weeks: int = 12) -> Optional[Path]:
        """
        Generate comprehensive performance dashboard.

        Shows accuracy, loss, and AUC trends over time in a single view.

        Args:
            weeks: Number of weeks to visualize

        Returns:
            Path to saved plot or None if failed
        """
        try:
            logger.info(f"Generating performance dashboard for last {weeks} weeks...")

            # Get training runs
            end_date = datetime.now()
            start_date = end_date - timedelta(weeks=weeks)

            training_runs = self.db.get_training_runs_by_date_range(
                start_date=start_date,
                end_date=end_date
            )

            if len(training_runs) == 0:
                logger.warning("No training runs found")
                return None

            # Extract metrics
            dates = [datetime.fromisoformat(run['run_timestamp']) for run in training_runs]
            train_accs = [run['train_accuracy'] for run in training_runs]
            val_accs = [run['val_accuracy'] for run in training_runs]
            train_losses = [run['train_loss'] for run in training_runs]
            val_losses = [run['val_loss'] for run in training_runs]
            roc_aucs = [run['roc_auc'] for run in training_runs]

            # Create dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(
                f'Continuous Learning Performance Dashboard\n'
                f'Last {weeks} weeks, {len(training_runs)} training runs',
                fontsize=16,
                fontweight='bold'
            )

            # 1. Accuracy over time
            axes[0, 0].plot(dates, train_accs, 'o-', label='Train Accuracy', linewidth=2, markersize=6)
            axes[0, 0].plot(dates, val_accs, 's-', label='Val Accuracy', linewidth=2, markersize=6)
            axes[0, 0].set_ylabel('Accuracy', fontsize=11)
            axes[0, 0].set_title('Model Accuracy Over Time', fontweight='bold')
            axes[0, 0].legend(loc='lower right')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim([0.5, 1.0])

            # 2. Loss over time
            axes[0, 1].plot(dates, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=6)
            axes[0, 1].plot(dates, val_losses, 's-', label='Val Loss', linewidth=2, markersize=6)
            axes[0, 1].set_ylabel('Loss', fontsize=11)
            axes[0, 1].set_title('Model Loss Over Time', fontweight='bold')
            axes[0, 1].legend(loc='upper right')
            axes[0, 1].grid(True, alpha=0.3)

            # 3. ROC AUC over time
            axes[1, 0].plot(dates, roc_aucs, 'o-', color='green', linewidth=2, markersize=6)
            axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Random')
            axes[1, 0].set_ylabel('ROC AUC', fontsize=11)
            axes[1, 0].set_xlabel('Date', fontsize=11)
            axes[1, 0].set_title('ROC AUC Over Time', fontweight='bold')
            axes[1, 0].legend(loc='lower right')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_ylim([0.5, 1.0])

            # 4. Sample counts
            n_train = [run['n_training_samples'] for run in training_runs]
            n_val = [run['n_validation_samples'] for run in training_runs]

            axes[1, 1].bar(dates, n_train, label='Train Samples', alpha=0.7, width=0.8)
            axes[1, 1].bar(dates, n_val, bottom=n_train, label='Val Samples', alpha=0.7, width=0.8)
            axes[1, 1].set_ylabel('Sample Count', fontsize=11)
            axes[1, 1].set_xlabel('Date', fontsize=11)
            axes[1, 1].set_title('Training Data Volume Over Time', fontweight='bold')
            axes[1, 1].legend(loc='upper left')
            axes[1, 1].grid(True, alpha=0.3, axis='y')

            # Format x-axis dates
            for ax in axes.flat:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)

            plt.tight_layout()

            # Save
            output_path = self.output_dir / f'performance_dashboard_{weeks}weeks.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Performance dashboard saved: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate performance dashboard: {e}", exc_info=True)
            return None

    def generate_all_visualizations(self, weeks: int = 12) -> Dict[str, Optional[Path]]:
        """
        Generate all visualizations at once.

        Args:
            weeks: Number of weeks to visualize

        Returns:
            Dict mapping visualization name to saved path
        """
        logger.info(f"Generating all visualizations for last {weeks} weeks...")

        results = {}

        # Temporal UMAP
        results['temporal_umap'] = self.generate_temporal_umap(weeks=weeks)

        # ROC progression
        results['roc_progression'] = self.generate_roc_progression(weeks=weeks)

        # Drift heatmap
        results['drift_heatmap'] = self.generate_drift_heatmap(weeks=weeks)

        # Performance dashboard
        results['performance_dashboard'] = self.generate_performance_dashboard(weeks=weeks)

        # Summary
        succeeded = sum(1 for path in results.values() if path is not None)
        logger.info(f"Generated {succeeded}/{len(results)} visualizations")

        return results
