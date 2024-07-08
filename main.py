import os
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Path to the directory containing this script
BASE_DIR = os.getcwd()  
MODELS_DIR = os.path.join(BASE_DIR, 'notebooks', 'models')  

def preprocess_image(img):
    img = image.load_img(img, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

model_path = os.path.join(MODELS_DIR, 'brain_tumor_detection_model.h5')
model = load_model(model_path)

# Function to predict using the loaded model
def predict_image_class(img_array, model):
    prediction = model.predict(img_array)
    return prediction[0]  
# Streamlit app
def main():
    st.title('Brain Tumor Detection')
    st.text('Upload a brain MRI image for prediction')

    # File uploader for image
    uploaded_file = st.file_uploader("Choose a brain MRI image ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded MRI Image.', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Predict the class
        prediction = predict_image_class(img_array, model)
        if prediction >= 0.5:
            st.write(f"Prediction: There is a tumor (Probability: {prediction[0]:.2f})")
        else:
            st.write(f"Prediction: No tumor (Probability: {prediction[0]:.2f})")

if __name__ == "__main__":
    main()
