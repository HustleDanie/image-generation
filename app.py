import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Page setup
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🦁 Animal Image Classifier")
st.write("Upload an image of an animal and get its predicted species using a Vision Transformer (ViT).")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Open and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Classifying... please wait..."):
            # Load model & processor
            model_name = "google/vit-base-patch16-224"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)

            # Preprocess input with explicit padding
            encoded = processor(images=[image], return_tensors="pt", padding=True)
            encoded = {k: v.cpu() for k, v in encoded.items()}
            model = model.cpu()

            # Inference
            with torch.no_grad():
                outputs = model(**encoded)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

            # Extract label and confidence
            label = model.config.id2label[predicted_class_idx]
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

        # Show results
        st.success(f"🦊 **Prediction:** {label}")
        st.info(f"📈 **Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
