import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image as PILImage
import tensorflow as tf

# Initialize model with input shape
@st.cache_resource
def load_emotion_model():
    try:
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
            *load_model('weights/emotion_model.h5').layers
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load model once
model = load_emotion_model()
if model is None:
    st.stop()

# Define class labels
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def predict_emotion(img):
    try:
        # Preprocess image
        img = img.convert('L')  # Convert to grayscale
        img = img.resize((48, 48))  # Resize image
        img_array = np.array(img) / 255.0  # Normalize
        img_array = img_array.reshape(1, 48, 48, 1)  # Reshape for model input
        
        # Make prediction
        pred = model.predict(img_array)
        predicted_class = np.argmax(pred)
        return emotion_labels[predicted_class]
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# UI Components
st.title("Emotion Recognition")
st.write("Upload an image to predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display image
        img = PILImage.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        # Predict emotion
        emotion = predict_emotion(img)
        if emotion:
            st.success(f"Predicted Emotion: {emotion}")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")