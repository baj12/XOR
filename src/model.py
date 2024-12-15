import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau


def build_and_train_model(hall_of_fame):
    best_individual = hall_of_fame[0]

    # Extract weights and biases from the best individual
    weights1, biases1, weights2, biases2, weights3, biases3 = best_individual

    # Build the model
    model = Sequential()
    model.add(Input(shape=(2,)))
    model.add(Dense(best_individual.hl1, activation='relu'))
    model.add(Dense(best_individual.hl2, activation='relu'))
    model.add(Dense(1, activation='sigmoid', dtype='float32'))

    # Select optimizer
    optimizer = best_individual.optimizer
    lr = best_individual.lr
    if optimizer == 'adam':
        opt = Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=lr)
    else:
        opt = Adam(learning_rate=lr)

    # Compile the model
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Set the model's weights and biases
    model.layers[1].set_weights([weights1, biases1])
    model.layers[3].set_weights([weights2, biases2])
    model.layers[5].set_weights([weights3, biases3])

    # Setup TensorBoard and Callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_final = f"logs/fit/{timestamp}"
    tensorboard_callback = TensorBoard(log_dir=log_dir_final, histogram_freq=1)
    early_stop_final = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr_final = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    history = model.fit(
        X_combined, y_combined,
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

    # Plot training accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'],
             label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'],
             label='Validation Accuracy', color='orange')
    plt.title('Best Model Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"plots/xor3_best_model_accuracy_{timestamp}.png")
    plt.show()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'],
             label='Validation Loss', color='orange')
    plt.title('Best Model Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f"plots/xor3_best_model_loss_{timestamp}.png")
    plt.show()
