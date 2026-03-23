import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np

# Import your model class
from src.model import HybridBlackgramNet

# --- Configuration ---
CLASS_NAMES = ['Anthracnose', 'Healthy', 'Leaf Crinckle', 'Powdery Mildew', 'Yellow Mosaic']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="QAgroNet Comparison", layout="wide")

# --- Safe Model Loading Function ---
@st.cache_resource
def load_model_safe(path, use_quantum):
    if not os.path.exists(path):
        return None
    
    try:
        # 1. Initialize the structure
        model = HybridBlackgramNet(num_classes=5, use_quantum=use_quantum)
        # 2. Load the weights (state_dict)
        state_dict = torch.load(path, map_location=DEVICE)
        
        # 3. Handle cases where the model was saved with 'model' key or as a full object
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            model.load_state_dict(state_dict['state_dict'])
        else:
            model.load_state_dict(state_dict)
            
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    return CLASS_NAMES[index], confidence.item() * 100

# --- UI Interface ---
st.title("🌱 Blackgram Disease Analysis: Quantum vs Classical")
st.markdown("---")

# Load models from the main D:\bld2 directory
model_qa = load_model_safe("model_qagronet.pth", use_quantum=True)
model_std = load_model_safe("model_standard.pth", use_quantum=False)

# Sidebar for Upload
st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Select a Leaf Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Layout: Image | Classical Result | Quantum Result
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader("Input Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("💻 Classical (ResNet50)")
        if model_std:
            label, conf = predict(model_std, img_tensor)
            st.info(f"Result: **{label}**")
            st.write(f"Confidence: {conf:.2f}%")
            st.progress(conf / 100)
        else:
            st.warning("`model_standard.pth` missing.")

    with col3:
        st.subheader("⚛️ QAgroNet (Hybrid)")
        if model_qa:
            label, conf = predict(model_qa, img_tensor)
            st.success(f"Result: **{label}**")
            st.write(f"Confidence: {conf:.2f}%")
            st.progress(conf / 100)
        else:
            st.warning("`model_qagronet.pth` missing.")
else:
    st.info("Please upload an image from the sidebar to begin analysis.")
