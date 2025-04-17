import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load models and data
@st.cache_resource
def load_models_and_data():
    # Load the Keras model
    keras_model = load_model(r"final_model.keras")
    
    # Load the LabelEncoder
    with open(r"label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    
    # Load the Random Forest model
    with open(r"random_forest_model.pkl", "rb") as file:
        rf_model = pickle.load(file)
    
    # Load the class labels
    with open(r"classes_list.pkl", "rb") as file:
        class_labels = pickle.load(file)
    
    # Load the nutritional data
    with open(r"unique_nutri_data.pkl", "rb") as file:
        nutri_data = pickle.load(file)
    
    return keras_model, label_encoder, rf_model, class_labels, nutri_data

# Load models and data
model, loaded_encoder, loaded_model, class_labels, nutri_data = load_models_and_data()

# Streamlit UI
st.title("Food Image Classification and Nutritional Analysis")

# Upload an image
uploaded_file = st.file_uploader("Upload an image of food", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    
    st.write(f"**Predicted Class:** {predicted_class_label}")
    
    # Get nutritional information
    nutri_info = nutri_data[nutri_data['label'] == predicted_class_label]
    
    if nutri_info.empty:
        st.write("No nutritional information available for this food item.")
    else:
        # Input desired weight
        desired_weight = st.number_input("Enter desired weight (in grams):", min_value=1, value=100)
        scale_factor = desired_weight / nutri_info["weight"].values[0]
        scaled_df = nutri_info.copy()
        columns_to_scale = ["calories", "protein", "carbohydrates", "fats", "fiber", "sugars", "sodium"]
        scaled_df[columns_to_scale] = np.ceil(nutri_info[columns_to_scale] * scale_factor)
        scaled_df["weight"] = desired_weight
        
        # Display scaled nutritional information
        st.write("**Nutritional Information (scaled):**")
        st.dataframe(scaled_df[["weight", "calories", "protein", "carbohydrates", "fats", "fiber", "sugars", "sodium"]])
        
        # Encode the label and prepare features for disease prediction
        scaled_df["label_encoded"] = loaded_encoder.transform(scaled_df["label"])
        features = ["weight", "calories", "protein", "carbohydrates", "fats", "fiber", "sugars", "sodium", "label_encoded"]
        X_new = scaled_df[features]
        
        # Predict disease suitability
        y_pred = loaded_model.predict(X_new)
        predictions = pd.DataFrame(y_pred, columns=["Diabetes", "Hypertension", "Heart Disease", "Kidney Disease"])
        
        st.write("**Disease Suitability Predictions:**")
        st.dataframe(predictions)
