import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Page config
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🦁 Animal Image Classifier")
st.write("Upload an image of an animal and get its predicted species using a Vision Transformer (ViT).")

# File upload
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Classifying... please wait..."):
            model_name = "google/vit-base-patch16-224"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)

            # Wrap image in list (simulate batch of 1)
            inputs = processor(images=[image], return_tensors="pt")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            model = model.to("cpu")

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()

            label = model.config.id2label[predicted_class_idx]
            confidence = torch.nn.functional.softmax(logits, dim=-1)[0][predicted_class_idx].item()

        st.success(f"🦊 **Prediction:** {label}")
        st.info(f"📈 **Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
