import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

# Page setup
st.set_page_config(page_title="Animal Classifier", layout="centered")
st.title("🦁 Animal Image Classifier")
st.write("Upload an image of an animal and get its predicted species using a Vision Transformer (ViT).")

# Upload
uploaded_file = st.file_uploader("📤 Upload an animal image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Classifying... please wait..."):
            # Load model and processor
            model_name = "google/vit-base-patch16-224"
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)

            # Preprocess (wrap in list for batch input)
            inputs = extractor(images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.cpu() for k, v in inputs.items()}
            model = model.cpu()

            # Inference
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

