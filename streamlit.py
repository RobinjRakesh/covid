import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Move this line inside the if block to avoid loading the model when not needed
# model_path = 'model.h5'

st.title("COVID-19 Identification Using CT Scan")
upload = st.file_uploader('Upload a CT scan image')

if upload is not None:
    file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    img = Image.open(upload)
    st.image(img, caption='Uploaded Image', width=300)

    if st.button('Predict'):
        model_path = 'model.h5'  # Load the model when the Predict button is pressed
        model = tf.keras.models.load_model(model_path)
        
        # Resize the image directly before prediction
        x = cv2.resize(opencv_image, (100, 100))
        x = np.expand_dims(x, axis=0)
        
        y = model.predict(x)
        ans = np.argmax(y, axis=1)
        
        # Define classes for better readability
        classes = ['COVID', 'Healthy', 'Other Pulmonary Disorder']
        st.title(classes[ans[0]])  # Display the predicted class

        # Optionally, you can also display the probability distribution
        st.write("Class Probabilities:")
        for i, prob in enumerate(y[0]):
            st.write(f"{classes[i]}: {prob:.2f}")
