# train.py
import tensorflow as tf
from model import create_model
from dataloader import preprocess_data
import matplotlib.pyplot as plt
import os

# Set your paths
TRAIN_DIR = 'facial-expression-dataset/train/train/'
TEST_DIR = 'facial-expression-dataset/test/test/'

def train_model():
    # Preprocess data
    x_train, x_test, y_train, y_test, le = preprocess_data(TRAIN_DIR, TEST_DIR)

    # Create and compile the model
    model = create_model()

    # Train the model
    history = model.fit(x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))

    # Plot results
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy Graph')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Loss Graph')
    plt.legend()

    plt.show()

    # Save the model
    model.save("emotion_model.h5")
    print("Model saved as emotion_model.h5")

if __name__ == '__main__':
    train_model()
