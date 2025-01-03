# model.py

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.utils import plot_model

from utils import Config

logger = logging.getLogger(__name__)
# tf.config.set_visible_devices([], 'GPU')


class CustomDebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        Custom callback to log detailed information at the end of each epoch.

        Parameters:
        - epoch (int): Current epoch number.
        - logs (dict): Dictionary of logs containing loss and accuracy metrics.
        """
        logger.debug(
            f"Epoch {epoch+1} - loss: {logs.get('loss'):.4f}, "
            f"accuracy: {logs.get('accuracy'):.4f}, "
            f"val_loss: {logs.get('val_loss'):.4f}, "
            f"val_accuracy: {logs.get('val_accuracy'):.4f}"
        )


def build_and_train_model(initial_weights, df: pd.DataFrame,
                          config: Config, X_train, X_val, y_train, y_val,
                          test_size: float = 0.2,
                          random_state: int = 42,
                          model_save_path: str = 'models/best_model.keras',
                          plot_accuracy_path: str = 'plots/best_model_accuracy.png',
                          plot_loss_path: str = 'plots/best_model_loss.png'):
    """
    Builds and trains a Keras model using the initial weights provided by the genetic algorithm.

    Parameters:
    - initial_weights (list): Flattened list of weights to initialize the model.
    - df (pd.DataFrame): DataFrame containing the dataset with 'x', 'y', 'label' columns.
    - config (Config): Configuration object containing model parameters.
    - test_size (float): Proportion of the dataset to include in the test split. (default: 0.2)
    - random_state (int): Controls the shuffling applied to the data before applying the split. (default: 42)
    - model_save_path (str): Filepath to save the trained model. (default: 'models/best_model.h5')
    - plot_accuracy_path (str): Filepath to save the accuracy plot. (default: 'plots/best_model_accuracy.png')
    - plot_loss_path (str): Filepath to save the loss plot. (default: 'plots/best_model_loss.png')
    """
    # Debug data information
    pid = os.getpid()
    logger.debug(f"{pid} DataFrame shape: {df.shape}")
    logger.debug(f"{pid} DataFrame columns: {df.columns.tolist()}")
    logger.debug(f"{pid} DataFrame head:\n{df.head()}")

    # Determine verbose level based on logger level
    log_level = logger.getEffectiveLevel()
    if log_level <= logging.DEBUG:
        verbose = 1
    elif log_level <= logging.INFO:
        verbose = 0
    else:
        verbose = 0

    # Validate input data
    if df.empty:
        logger.error(f"{pid} Empty DataFrame provided")
        raise ValueError("Empty DataFrame")

    if not all(col in df.columns for col in ['x', 'y', 'label']):
        logger.error(f"{pid} Missing required columns")
        raise ValueError("DataFrame must contain 'x', 'y', 'label' columns")

    # Debug initial weights
    logger.debug(f"{pid} Initial weights length: {len(initial_weights)}")
    logger.debug(
        f"{pid} Initial weights range: [{min(initial_weights)}, {max(initial_weights)}]")

    # Split the data into features and labels
    X = df[['x', 'y']].values
    y = df['label'].values

    logger.debug(f"{pid} Features shape: {X.shape}")
    logger.debug(f"{pid} Labels shape: {y.shape}")

    # Create output directories
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_accuracy_path).parent.mkdir(parents=True, exist_ok=True)

    # Build and configure model with debug info
    model = build_model(config)
    logger.debug(f"{pid} Model summary:\n{model.get_config()}")

    # Add debug callbacks
    callbacks = [
        TensorBoard(log_dir='./logs', histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=10, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                          patience=5, verbose=verbose),
        CustomDebugCallback()  # Custom callback for detailed monitoring
    ]

    # Set the initial weights from the genetic algorithm
    try:
        weight_tuples = []
        idx = 0
        for layer in model.layers:
            weights_shape = layer.get_weights()
            if weights_shape:
                weight_shape = weights_shape[0].shape
                bias_shape = weights_shape[1].shape
                weight_size = np.prod(weight_shape)
                bias_size = np.prod(bias_shape)

                weights = np.array(
                    initial_weights[idx:idx+weight_size]).reshape(weight_shape)
                weight_tuples.append(weights)
                idx += weight_size

                biases = np.array(
                    initial_weights[idx:idx+bias_size]).reshape(bias_shape)
                weight_tuples.append(biases)
                idx += bias_size
        model.set_weights(weight_tuples)
    except Exception as e:
        logger.error(f"Error setting initial weights: {e}")
        return

    # Compile the model with a chosen optimizer
    # You can modify or parameterize this
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Setup TensorBoard and Callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_final = f"logs/fit/{timestamp}"
    tensorboard_callback = TensorBoard(log_dir=log_dir_final, histogram_freq=1)
    early_stop_final = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr_final = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Ensure directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_accuracy_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_loss_path), exist_ok=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[tensorboard_callback, early_stop_final, reduce_lr_final],
        verbose=verbose
    )
    logger.debug(f"{pid} Model training completed")

    model.save(model_save_path)
    logger.info(f"{pid} Model saved to {model_save_path}")

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.savefig(plot_accuracy_path)
    plt.close()
    logger.info(f"{pid} Accuracy plot saved to {plot_accuracy_path}")

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(plot_loss_path)
    plt.close()
    logger.info(f"{pid} Loss plot saved to {plot_loss_path}")

    # Return the trained model
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using various metrics.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


def initialize_tensorflow():
    """
    Initialize TensorFlow to preload necessary components.
    This can help prevent delays during the first model operation.
    """
    try:
        logger.debug("Initializing TensorFlow...")
        # Perform a simple TensorFlow operation
        tf.constant(1)
        logger.debug("TensorFlow initialized successfully.")
    except Exception as e:
        logger.error(f"TensorFlow initialization failed: {e}")
        raise


def validate_config(config):
    assert isinstance(
        config.model.hl1, int) and config.model.hl1 > 0, "hl1 must be a positive integer."
    assert isinstance(
        config.model.hl2, int) and config.model.hl2 > 0, "hl2 must be a positive integer."
    valid_activations = ['relu', 'sigmoid', 'tanh', 'softmax', 'linear']
    assert config.model.activation in valid_activations, f"Unsupported activation function: {config.model.activation}"
    logger.debug("Configuration parameters validated successfully.")


def build_model(config: Config) -> Sequential:
    """
    Builds and returns a Keras Sequential model using configuration parameters.

    Parameters:
    - config (Config): Configuration object containing model parameters.

    Returns:
    - model (Sequential): Compiled Keras model.
    """
    pid = os.getpid()
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = uuid.uuid4()

    # Disable GPU (for testing purposes)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # logger.debug(f"{pid} GPU disabled for this process.")
    initialize_tensorflow()

    logger.debug(f"{pid} - Model configuration: {config.model}")

    model = Sequential(name=f"model_{pid}_{current_date}_{unique_id}")
    logger.debug(f"{pid} Sequential done.")

    model.add(Input(shape=(2,)))  # Input layer matching feature dimensions
    logger.debug(f"{pid} add done.")

    # Validate configuration parameters
    validate_config(config)

    logger.debug(
        f"{pid} add 2 {config.model.hl1} - {config.model.activation}.")
    # Add first hidden layer
    model.add(Dense(
        units=config.model.hl1,
        activation=config.model.activation,
        name='hidden_layer_1'
    ))
    logger.debug(
        f"{pid} Added hidden_layer_1 with {config.model.hl1} units and '{config.model.activation}' activation.")

    # Add second hidden layer
    model.add(Dense(
        units=config.model.hl2,
        activation=config.model.activation,
        name='hidden_layer_2'
    ))
    logger.debug(
        f"{pid} Added hidden_layer_2 with {config.model.hl2} units and '{config.model.activation}' activation.")

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid', name='output_layer'))
    logger.debug(
        f"{pid} Added output_layer with 1 unit and 'sigmoid' activation.")

    # Configure optimizer
    optimizer = get_optimizer(config.model.optimizer, config.model.lr)
    logger.debug(
        f"{pid} Configured optimizer: {config.model.optimizer} with learning rate {config.model.lr}")

    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    logger.debug(f"{pid} Model compiled successfully.")
    # Setup TensorBoard callback for profiling
    log_dir = "logs/profile/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Adjust batch numbers as needed
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, profile_batch='500,520')

    # # Visualize the model architecture
    # try:
    #     plot_model(model, to_file=f"plots/model_architecture.{pid}.{current_date}.{unique_id}.png",
    #                show_shapes=True, show_layer_names=True)
    #     logger.debug(
    #         f"{pid} Model architecture saved to 'model_architecture.png'.")
    # except Exception as e:
    #     logger.error(f"{pid} Failed to plot model architecture: {e}")
    #     raise

    return model


def get_optimizer(name: str, lr: float):
    """
    Returns a Keras optimizer based on the given name and learning rate.

    Parameters:
    - name (str): Name of the optimizer ('adam', 'sgd', 'rmsprop', etc.).
    - lr (float): Learning rate for the optimizer.

    Returns:
    - optimizer: Keras optimizer instance.
    """
    if name.lower() == 'adam':
        return Adam(learning_rate=lr)
    elif name.lower() == 'sgd':
        return SGD(learning_rate=lr)
    elif name.lower() == 'rmsprop':
        return RMSprop(learning_rate=lr)
    else:
        logger.warning(
            f"Optimizer '{name}' not recognized. Using 'adam' as default.")
        return Adam(learning_rate=lr)
