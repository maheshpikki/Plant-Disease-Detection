import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = tf.keras.models.load_model('trained_plant_disease_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))  # Resize image to 128x128
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def predict_disease(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    result_index = np.argmax(predictions)
    return result_index

# Function to predict fertilizer based on disease
def predict_fertilizer(disease):
    # Define a mapping between disease and fertilizer
    fertilizer_mapping = {
        'Tomato___Bacterial_spot': 'Ammonium Nitrate',
        'Tomato___Early_blight': 'Triple Superphosphate',
        'Tomato___Late_blight': 'Calcium Nitrate',
        'Tomato___Leaf_mold': 'Potassium Sulfate',
        'Tomato___Septoria_leaf_spot': 'Potassium Nitrate',
        'Tomato___Spider_mites': 'Urea',
        'Tomato___Target_spot': 'Diammonium Phosphate',
        'Tomato___Yellow_leaf_curl_virus': 'Magnesium Sulfate',
        'Potato___Bacterial_spot': 'Sulphate of Potash',
        'Potato___Early_blight': 'Ammonium Phosphate',
        'Potato___Late_blight': 'Potassium Chloride',
        'Potato___Leaf_mold': 'Urea',
        'Potato___Septoria_leaf_spot': 'Calcium Ammonium Nitrate',
        'Potato___Spider_mites': 'Monoammonium Phosphate',
        'Potato___Target_spot': 'Ammonium Sulfate',
        'Potato___Yellow_leaf_curl_virus': 'Potassium Sulfate',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Nitrogen, Phosphorus, Potassium (NPK) Fertilizer',
        'Corn___Bacterial_spot': 'Urea',
        'Corn___Early_blight': 'Ammonium Nitrate',
        'Corn___Late_blight': 'Potassium Sulfate',
        'Corn___Leaf_mold': 'Calcium Nitrate',
        'Corn___Septoria_leaf_spot': 'Triple Superphosphate',
        'Corn___Spider_mites': 'Diammonium Phosphate',
        'Corn___Target_spot': 'Magnesium Sulfate',
        'Corn___Yellow_leaf_curl_virus': 'Sulphate of Potash',
    }
    # Lookup the fertilizer based on the predicted disease
    return fertilizer_mapping.get(disease, 'Unknown fertilizer')

# Streamlit App
def main():
    st.title("Plant Disease Detection App")

    # Option to upload image
    uploaded_file = st.file_uploader("Upload Plant Image", type=["jpg", "jpeg", "png"])

    # Load the validation set to access class names
    validation_set = tf.keras.utils.image_dataset_from_directory(
        'valid',
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=32,
        image_size=(128, 128),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False
    )
    class_names = validation_set.class_names

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Center the image using columns
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction and display result
        if st.button("Classify"):
            st.write("Classifying...")
            prediction = predict_disease(image)
            class_name = class_names[prediction]
            st.success(f"Predicted Disease: {class_name}")

            # Predict fertilizer based on the disease
            fertilizer = predict_fertilizer(class_name)
            st.success(f"Suggested Fertilizer: {fertilizer}")

if __name__ == '__main__':
    main()
