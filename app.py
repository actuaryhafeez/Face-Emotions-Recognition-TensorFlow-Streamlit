import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.image import resize

# Load the trained model
model = load_model('models/imageclassifier.h5')

# Set up the Streamlit app
st.title("Emotion Recognition App")
st.write("Upload an image and let the model predict the emotion!")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Read and preprocess the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
    resized_image = resize(image, (256, 256))
    normalized_image = resized_image / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0)
    
    # Make prediction
    prediction = model.predict(expanded_image)
    
    # Interpret prediction
    if prediction > 0.5:
        emotion = "Sad ☹️"
    else:
        emotion = "Happy 🤗"
    
    # Display results
    image_width = 400  # Set the desired width for the displayed image
    st.image(image, caption='Uploaded Image', width=image_width)
    st.write(f"Predicted Emotion: {emotion}")
