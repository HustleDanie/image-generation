import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Animal Classifier")
st.title("🦁 Animal Image Classifier")
st.write("Upload an animal image to identify its species using a Vision Transformer.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # 🛠 Force resize the image to avoid padding-related tensor errors
        image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("🔍 Classifying..."):
            classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
            result = classifier(image)

        label = result[0]["label"]
        confidence = result[0]["score"]

        st.success(f"🦊 **Prediction:** {label}")
        st.info(f"📈 **Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
