"""
Incremental Model Trainer for Continuous Learning

Trains and updates neural network models incrementally as new data arrives.

Key Features:
- Sliding window training (last N weeks of data)
- Incremental updates without full retraining
- Model versioning and comparison
- Performance tracking over time
- Automatic model selection (best performing)

Training Modes:
1. Incremental: Update existing model with new data
2. Full Retrain: Train from scratch on sliding window
3. Hybrid: Incremental + periodic full retrains

Usage:
    from continuous.incremental_trainer import IncrementalTrainer

    trainer = IncrementalTrainer(config, db_path, model_dir)
    result = trainer.train_incremental(weeks=4)
    trainer.evaluate_latest_model()
"""

import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

from .feature_database import FeatureDatabase

logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Track training session metrics"""

    def __init__(self):
        self.sessions = []

    def record_session(self,
                      session_type: str,
                      n_samples: int,
                      train_acc: float,
                      val_acc: float,
                      test_acc: float,
                      duration_seconds: float):
        """Record a training session"""
        self.sessions.append({
            'timestamp': datetime.now(),
            'session_type': session_type,
            'n_samples': n_samples,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'test_accuracy': test_acc,
            'duration_seconds': duration_seconds
        })

    def get_recent_sessions(self, n: int = 10) -> List[Dict]:
        """Get N most recent training sessions"""
        return self.sessions[-n:]


class IncrementalTrainer:
    """
    Incremental trainer for continuous learning.

    Handles:
    - Loading data from sliding time window
    - Incremental model updates
    - Full retraining when needed
    - Model versioning and selection
    - Performance monitoring
    """

    def __init__(self,
                 config,
                 db_path: Path,
                 model_dir: Path):
        """
        Initialize incremental trainer.

        Args:
            config: Configuration object with model parameters
            db_path: Path to feature database
            model_dir: Directory for saving models
        """
        self.config = config
        self.db = FeatureDatabase(db_path)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = TrainingMetrics()
        self.scaler = None
        self.current_model = None

        logger.info(f"IncrementalTrainer initialized")
        logger.info(f"Database: {db_path}")
        logger.info(f"Model directory: {model_dir}")

    def load_training_data(self,
                          weeks: int = 4,
                          max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from sliding time window.

        Args:
            weeks: Number of weeks to include (sliding window)
            max_samples: Optional limit on samples (None = all)

        Returns:
            Tuple of (X, y) with features and labels
        """
        logger.info(f"Loading training data: last {weeks} weeks")

        end_date = datetime.now()
        start_date = end_date - timedelta(weeks=weeks)

        X, y = self.db.load_features_by_date_range(
            start_date,
            end_date,
            max_samples=max_samples
        )

        logger.info(f"Loaded {len(X)} samples")
        logger.info(f"Class distribution: {np.sum(y==0)} negative, {np.sum(y==1)} positive")

        return X, y

    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, val_size: float = 0.1):
        """
        Prepare data for training: normalize and split.

        Args:
            X: Feature matrix
            y: Labels
            test_size: Fraction for test set
            val_size: Fraction of remaining for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        """
        # Split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Split train+val into train and val
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_fraction, random_state=42, stratify=y_trainval
        )

        # Normalize features
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
        else:
            # Use existing scaler (incremental learning)
            X_train = self.scaler.transform(X_train)

        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler

    def build_model(self, input_dim: int) -> keras.Model:
        """
        Build neural network model.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras model
        """
        # Get architecture from config or use defaults
        hidden_layers = getattr(self.config, 'hidden_layers', [128, 64, 32])
        activation = getattr(self.config, 'activation', 'relu')
        learning_rate = getattr(self.config, 'learning_rate', 0.001)

        # Build model
        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(input_dim,)))

        for units in hidden_layers:
            model.add(keras.layers.Dense(units, activation=activation))
            model.add(keras.layers.Dropout(0.3))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )

        logger.info(f"Model built: {len(hidden_layers)} hidden layers, {sum(hidden_layers)} total units")

        return model

    def train_full(self,
                  weeks: int = 4,
                  epochs: int = 20,
                  batch_size: int = 32,
                  max_samples: Optional[int] = None) -> Dict:
        """
        Full training from scratch on sliding window data.

        Args:
            weeks: Sliding window size
            epochs: Training epochs
            batch_size: Batch size
            max_samples: Optional sample limit

        Returns:
            Dict with training results
        """
        logger.info("=== Full Training ===")
        start_time = datetime.now()

        # Load data
        X, y = self.load_training_data(weeks=weeks, max_samples=max_samples)

        if len(X) < 100:
            raise ValueError(f"Insufficient data: {len(X)} samples (need at least 100)")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.prepare_data(X, y)

        # Build new model
        input_dim = X_train.shape[1]
        model = self.build_model(input_dim)

        # Train
        logger.info(f"Training for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ]
        )

        # Evaluate
        train_loss, train_acc, train_auc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Training complete in {duration:.1f}s")
        logger.info(f"Train accuracy: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")

        # Save model
        model_path = self.model_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        model.save(model_path)
        logger.info(f"Model saved: {model_path}")

        # Save scaler
        scaler_path = self.model_dir / f"scaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Update current model
        self.current_model = model
        self.scaler = scaler

        # Record metrics
        self.metrics.record_session(
            session_type='full',
            n_samples=len(X),
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc,
            duration_seconds=duration
        )

        # Store in database
        self.db.insert_training_run({
            'run_timestamp': datetime.now(),
            'model_path': str(model_path),
            'data_start_date': datetime.now() - timedelta(weeks=weeks),
            'data_end_date': datetime.now(),
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'roc_auc': float(test_auc),
            'update_mode': 'full',
            'duration_seconds': duration
        })

        return {
            'status': 'success',
            'model_path': str(model_path),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'n_samples': len(X),
            'duration_seconds': duration
        }

    def train_incremental(self,
                         weeks: int = 1,
                         epochs: int = 5,
                         batch_size: int = 32) -> Dict:
        """
        Incremental training: update existing model with new data.

        Args:
            weeks: How many weeks of new data to train on
            epochs: Fine-tuning epochs
            batch_size: Batch size

        Returns:
            Dict with training results
        """
        logger.info("=== Incremental Training ===")

        if self.current_model is None:
            logger.warning("No existing model, performing full training instead")
            return self.train_full(weeks=4, epochs=20, batch_size=batch_size)

        start_time = datetime.now()

        # Load recent data
        X, y = self.load_training_data(weeks=weeks)

        if len(X) < 50:
            logger.warning(f"Insufficient new data: {len(X)} samples")
            return {'status': 'skipped', 'reason': 'insufficient_data'}

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, _ = self.prepare_data(X, y)

        # Fine-tune existing model
        logger.info(f"Fine-tuning for {epochs} epochs...")
        history = self.current_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        # Evaluate
        train_loss, train_acc, train_auc = self.current_model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc, val_auc = self.current_model.evaluate(X_val, y_val, verbose=0)
        test_loss, test_acc, test_auc = self.current_model.evaluate(X_test, y_test, verbose=0)

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(f"Incremental training complete in {duration:.1f}s")
        logger.info(f"Test accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}")

        # Save updated model
        model_path = self.model_dir / f"model_incremental_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
        self.current_model.save(model_path)

        # Record metrics
        self.metrics.record_session(
            session_type='incremental',
            n_samples=len(X),
            train_acc=train_acc,
            val_acc=val_acc,
            test_acc=test_acc,
            duration_seconds=duration
        )

        # Store in database
        self.db.insert_training_run({
            'run_timestamp': datetime.now(),
            'model_path': str(model_path),
            'data_start_date': datetime.now() - timedelta(weeks=weeks),
            'data_end_date': datetime.now(),
            'n_training_samples': len(X_train),
            'n_validation_samples': len(X_val),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'test_accuracy': float(test_acc),
            'roc_auc': float(test_auc),
            'update_mode': 'incremental',
            'duration_seconds': duration
        })

        return {
            'status': 'success',
            'model_path': str(model_path),
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'n_samples': len(X),
            'duration_seconds': duration
        }

    def load_latest_model(self) -> Optional[keras.Model]:
        """
        Load most recent model from model directory.

        Returns:
            Loaded Keras model or None
        """
        model_files = sorted(self.model_dir.glob('model_*.keras'))

        if not model_files:
            logger.warning("No saved models found")
            return None

        latest_model = model_files[-1]
        logger.info(f"Loading model: {latest_model}")

        model = keras.models.load_model(latest_model)
        self.current_model = model

        # Load corresponding scaler
        scaler_files = sorted(self.model_dir.glob('scaler_*.pkl'))
        if scaler_files:
            latest_scaler = scaler_files[-1]
            with open(latest_scaler, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler: {latest_scaler}")

        return model

    def evaluate_on_recent_data(self, weeks: int = 1) -> Dict:
        """
        Evaluate current model on recent data.

        Args:
            weeks: How many weeks of recent data to evaluate on

        Returns:
            Dict with evaluation metrics
        """
        if self.current_model is None:
            self.load_latest_model()

        if self.current_model is None:
            raise ValueError("No model available for evaluation")

        logger.info(f"Evaluating on last {weeks} weeks of data")

        # Load recent data
        X, y = self.load_training_data(weeks=weeks)

        # Normalize
        X_norm = self.scaler.transform(X)

        # Evaluate
        loss, accuracy, auc = self.current_model.evaluate(X_norm, y, verbose=0)

        logger.info(f"Evaluation: accuracy={accuracy:.4f}, AUC={auc:.4f}")

        return {
            'accuracy': float(accuracy),
            'auc': float(auc),
            'n_samples': len(X)
        }

    def get_training_history(self, n_sessions: int = 10) -> List[Dict]:
        """Get recent training session history"""
        return self.metrics.get_recent_sessions(n_sessions)
