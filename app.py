
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
import requests

# Load the pre-trained models
@st.cache_resource
def load_image_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
    model.load_state_dict(torch.load("image_classification_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_text_model():
    return load_model("text_classification_model.h5")

# Load tokenizer and embedding matrix (if needed)
tokenizer = joblib.load("tokenizer.joblib")  # Ensure this file exists

# Define categories for image classification
categories = ['Sports', 'News', 'Sci/Tech', 'Entertainment']

# Preprocessing for image classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Preprocessing for text classification
max_length = 100
def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_length)
    return padded_seq

# Function to classify images
def classify_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)

    # Fetch the ImageNet labels
    labels_url = "https://raw.githubusercontent.com/JaswanthRemiel/ImageClassificationLABELS/refs/heads/main/imagenet-simple-labels.json"
    labels = requests.get(labels_url).json()
    predicted_label = labels[predicted_idx.item()]

    # Map the predicted index to one of the predefined categories
    mapped_category = categories[predicted_idx.item() % 4]

    return f"Predicted Label: {predicted_label}", f"Mapped Category: {mapped_category}"

# Function to classify text
def classify_text(text, model):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_mapping = {0: 'Sports', 1: 'News', 2: 'Sci/Tech', 3: 'Entertainment'}  # Update with actual classes
    return class_mapping[predicted_class]

# Streamlit App
st.title("Image and Text Classification App")

option = st.selectbox("Choose a task:", ("Image Classification", "Text Classification"))

if option == "Image Classification":
    st.header("Upload an image for classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        if st.button("Classify Image"):
            model = load_image_model()
            label, category = classify_image(uploaded_file, model)
            st.write(label)
            st.write(category)

elif option == "Text Classification":
    st.header("Enter text for classification")
    user_input = st.text_area("Type your text here:")
    if st.button("Classify Text"):
        if user_input:
            model = load_text_model()
            result = classify_text(user_input, model)
            st.write(f"Predicted Class: {result}")
        else:
            st.warning("Please enter some text.")
