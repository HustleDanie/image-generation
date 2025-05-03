import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Title and description
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🦁 Animal Image Classifier")
st.write("Upload an image of an animal and get its predicted species using a Vision Transformer (ViT).")

# Image upload
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load model and processor
    with st.spinner("🔍 Classifying... please wait..."):
        model_name = "google/vit-base-patch16-224"
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForImageClassification.from_pretrained(model_name)

        # Preprocess and predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        # Show results
        label = model.config.id2label[predicted_class_idx]
        confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

    st.success(f"🦊 **Prediction:** {label}")
    st.info(f"📈 **Confidence:** {confidence:.2%}")
