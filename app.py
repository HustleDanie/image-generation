import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Set up the Streamlit page
st.set_page_config(page_title="Animal Image Classifier", layout="centered")
st.title("🦁 Animal Image Classifier")
st.write("Upload an image of an animal, and the model will predict its species using a Vision Transformer (ViT).")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Open and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Classifying... please wait..."):
            # Load the model and feature extractor
            model_name = "google/vit-base-patch16-224"
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)

            # Preprocess the image
            inputs = extractor(images=image, return_tensors="pt")
            inputs = {k: v.cpu() for k, v in inputs.items()}
            model = model.cpu()

            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

            # Retrieve the label and confidence
            label = model.config.id2label[predicted_class_idx]
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

        # Display the results
        st.success(f"🦊 **Prediction:** {label}")
        st.info(f"📈 **Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
