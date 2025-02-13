import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import Add, Concatenate, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Import the config classes
from utils import Config, ModelConfig

logger = logging.getLogger(__name__)


def get_optimizer(optimizer_name: str, learning_rate: float):
    """
    Get the specified optimizer with the given learning rate.

    Parameters:
    - optimizer_name (str): Name of the optimizer ('adam', 'sgd', or 'rmsprop')
    - learning_rate (float): Learning rate for the optimizer

    Returns:
    - tf.keras.optimizers: The specified optimizer
    """
    optimizers = {
        'adam': Adam(learning_rate=learning_rate),
        'sgd': SGD(learning_rate=learning_rate),
        'rmsprop': RMSprop(learning_rate=learning_rate)
    }

    return optimizers.get(optimizer_name.lower(), Adam(learning_rate=learning_rate))


def build_model(config: ModelConfig):
    """
    Build model with optional skip connections.

    Parameters:
    - config (ModelConfig): Model configuration containing architecture details

    Returns:
    - tf.keras.Model: Compiled Keras model
    """
    inputs = Input(shape=(config.input_dim,))
    x = inputs

    # Add hidden layers
    previous_layers = []
    for i, units in enumerate(config.hidden_layers):
        x = Dense(
            units=units,
            activation=config.activation,
            name=f'dense_{i}'
        )(x)
        previous_layers.append(x)

        # Add skip connections if specified
        if config.skip_connections:
            if config.skip_connections == 'residual' and i > 0:
                x = Add()([x, previous_layers[-2]])
            elif config.skip_connections == 'dense' and i > 0:
                x = Concatenate()([x] + previous_layers[:-1])

    # Output layer
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile model
    optimizer = get_optimizer(config.optimizer, config.lr)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


class CustomDebugCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed monitoring during training."""

    def on_epoch_end(self, epoch, logs=None):
        logger.debug(f"Epoch {epoch + 1} - loss: {logs['loss']:.4f}, "
                     f"accuracy: {logs['accuracy']:.4f}, "
                     f"val_loss: {logs['val_loss']:.4f}, "
                     f"val_accuracy: {logs['val_accuracy']:.4f}")


def build_and_train_model(initial_weights, df, config: Config,
                          X_train, X_val, y_train, y_val,
                          model_save_path='models/best_model.keras',
                          plot_accuracy_path='plots/accuracy.png',
                          plot_loss_path='plots/loss.png'):
    """
    Build and train a model using the provided configuration and data.

    Parameters:
    - initial_weights: Initial weights from genetic algorithm
    - df: DataFrame containing the dataset
    - config: Configuration object
    - X_train, X_val: Training and validation features
    - y_train, y_val: Training and validation labels
    - model_save_path: Path to save the trained model
    - plot_accuracy_path: Path to save accuracy plot
    - plot_loss_path: Path to save loss plot

    Returns:
    - trained model
    """
    # Set up directories
    for path in [model_save_path, plot_accuracy_path, plot_loss_path]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Build model
    model = build_model(config.model)

    # Debug information
    logger.debug(f"Model summary:\n{model.summary()}")

    # Set up callbacks
    callbacks = [
        TensorBoard(log_dir='logs/fit', histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=5, verbose=1),
        CustomDebugCallback()
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=config.ga.epochs,
        batch_size=config.model.batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # Save model
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # Plot training history
    plot_training_history(history, plot_accuracy_path, plot_loss_path)

    return model


def plot_training_history(history, accuracy_path, loss_path):
    """Plot and save training history."""
    import matplotlib.pyplot as plt

    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(accuracy_path)
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_path)
    plt.close()


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using test data.

    Parameters:
    - model: Trained Keras model
    - X_test: Test features
    - y_test: Test labels

    Returns:
    - dict: Evaluation metrics
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    predictions = model.predict(X_test)

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': predictions
    }
