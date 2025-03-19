import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource()  # Cache the model to avoid reloading on every run
def load_model():
    return tf.keras.models.load_model("inception_model.h5")

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((299, 299))  # Resize to match InceptionV3 input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.title("🍃 Plant Disease Detection 🌿")
st.write("Upload an image to classify it as **Fresh** or **Defective**.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)[0][0]  # Assuming binary classification

    # Set confidence and label
    if prediction > 0.5:
        label = "✅ Fresh"
        color = "green"
    else:
        label = "❌ Defect"
        color = "red"

    # Display results with styling
    st.markdown(f"### **Prediction: <span style='color:{color}'>{label}</span>**", unsafe_allow_html=True)
    st.write(f"🧪 **Confidence Score:** `{prediction:.4f}`")

