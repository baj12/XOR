# model.py

import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
import pandas as pd

def build_and_train_model(hall_of_fame, df: pd.DataFrame, 
                          test_size: float = 0.2, 
                          random_state: int = 42,
                          model_save_path: str = 'models/best_model.h5',
                          plot_accuracy_path: str = 'plots/best_model_accuracy.png',
                          plot_loss_path: str = 'plots/best_model_loss.png'):
    """
    Builds and trains a Keras model using the best individual from the genetic algorithm.

    Parameters:
    - hall_of_fame (deap.tools.HallOfFame): Contains the best individuals from the genetic algorithm.
    - df (pd.DataFrame): DataFrame containing the dataset with 'x', 'y', 'label' columns.
    - test_size (float): Proportion of the dataset to include in the test split. (default: 0.2)
    - random_state (int): Controls the shuffling applied to the data before applying the split. (default: 42)
    - model_save_path (str): Filepath to save the trained model. (default: 'models/best_model.h5')
    - plot_accuracy_path (str): Filepath to save the accuracy plot. (default: 'plots/best_model_accuracy.png')
    - plot_loss_path (str): Filepath to save the loss plot. (default: 'plots/best_model_loss.png')
    """
    # Extract the best individual
    best_individual = hall_of_fame[0]
    
    # Assuming the best_individual has attributes hl1, hl2, optimizer, lr
    # If best_individual is a list, adjust attribute access accordingly
    # Example: [hl1, hl2, optimizer, lr]
    try:
        hl1 = best_individual[0]
        hl2 = best_individual[1]
        optimizer_choice = best_individual[2]
        learning_rate = best_individual[3]
    except (IndexError, TypeError):
        raise ValueError("Best individual does not have the required attributes: hl1, hl2, optimizer, lr.")
    
    # Split the data into features and labels
    X = df[['x', 'y']].values
    y = df['label'].values
    
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Build the model
    model = Sequential()
    model.add(Input(shape=(2,)))  # Input layer matching feature dimensions
    model.add(Dense(hl1, activation='relu'))
    model.add(Dense(hl2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    
    # Select optimizer based on the individual's choice
    optimizer = select_optimizer(optimizer_choice, learning_rate)
    
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Setup TensorBoard and Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_final = f"logs/fit/{timestamp}"
    tensorboard_callback = TensorBoard(log_dir=log_dir_final, histogram_freq=1)
    early_stop_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_accuracy_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_loss_path), exist_ok=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[tensorboard_callback, early_stop_final, reduce_lr_final],
        verbose=1
    )
    
    # Evaluate the model
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Best Model Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(plot_accuracy_path)
    plt.close()
    print(f"Accuracy plot saved to {plot_accuracy_path}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Best Model Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(plot_loss_path)
    plt.close()
    print(f"Loss plot saved to {plot_loss_path}")

def select_optimizer(choice: str, lr: float):
    """
    Selects and returns the optimizer based on the choice and learning rate.

    Parameters:
    - choice (str): Optimizer type ('adam', 'sgd', 'rmsprop').
    - lr (float): Learning rate.

    Returns:
    - optimizer (keras.optimizers.Optimizer): Configured optimizer.
    """
    if choice.lower() == 'adam':
        return Adam(learning_rate=lr)
    elif choice.lower() == 'sgd':
        return SGD(learning_rate=lr)
    elif choice.lower() == 'rmsprop':
        return RMSprop(learning_rate=lr)
    else:
        print(f"Unknown optimizer choice '{choice}'. Defaulting to Adam.")
        return Adam(learning_rate=lr)