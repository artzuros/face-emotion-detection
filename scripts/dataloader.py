# dataloader.py
import os
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tqdm import tqdm

def load_dataset(directory):
    """Load images and their corresponding labels from the directory."""
    image_paths = []
    labels = []
    
    for label in os.listdir(directory):
        for filename in os.listdir(directory+label):
            image_path = os.path.join(directory, label, filename)
            image_paths.append(image_path)
            labels.append(label)
        print(f"Completed {label}")
        
    return image_paths, labels

def extract_features(images):
    """Extract and preprocess images for model input."""
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

def preprocess_data(train_dir, test_dir, num_classes=7):
    """Load and preprocess the dataset."""
    # Load training and test dataset
    train_image_paths, train_labels = load_dataset(train_dir)
    test_image_paths, test_labels = load_dataset(test_dir)
    
    # Create dataframes
    train = pd.DataFrame({'image': train_image_paths, 'label': train_labels})
    test = pd.DataFrame({'image': test_image_paths, 'label': test_labels})
    
    # Shuffle the dataset
    train = train.sample(frac=1).reset_index(drop=True)
    
    # Feature extraction
    train_features = extract_features(train['image'])
    test_features = extract_features(test['image'])
    
    # Normalize images
    x_train = train_features / 255.0
    x_test = test_features / 255.0
    
    # Encode labels
    le = LabelEncoder()
    le.fit(train['label'])
    y_train = le.transform(train['label'])
    y_test = le.transform(test['label'])
    
    # One-hot encoding of labels
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    return x_train, x_test, y_train, y_test, le
