# test.py
import numpy as np
from keras.models import load_model
import random
import matplotlib.pyplot as plt
from dataloader import preprocess_data

def test_model():
    # Preprocess data
    x_train, x_test, y_train, y_test, le = preprocess_data('facial-expression-dataset/train/train/', 'facial-expression-dataset/test/test/')

    # Load the trained model
    model = load_model('emotion_model.h5')

    # Test the model with random image
    image_index = random.randint(0, len(x_test))
    print("Original Output:", le.inverse_transform([y_test[image_index].argmax()])[0])

    pred = model.predict(x_test[image_index].reshape(1, 48, 48, 1))
    prediction_label = le.inverse_transform([pred.argmax()])[0]
    print("Predicted Output:", prediction_label)

    plt.imshow(x_test[image_index].reshape(48, 48), cmap='gray')
    plt.show()

if __name__ == '__main__':
    test_model()
