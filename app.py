
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================
# PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ==================================
# LOAD MODEL
# ==================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# ==================================
# CLASS NAMES
# ==================================

classes = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

# ==================================
# MAIN TITLE
# ==================================

st.title("🧠 Brain Tumor Classification")
st.write(
    "Upload an MRI brain image to get an experimental "
    "AI classification."
)

st.divider()

# ==================================
# IMAGE UPLOAD
# ==================================

uploaded_file = st.file_uploader(
    "Upload Brain MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.subheader("Uploaded MRI Image")
        st.image(image, use_container_width=True)

    # ==================================
    # PREPROCESS IMAGE
    # ==================================

    img = image.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)

    # Assumes training used rescale=1./255
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ==================================
    # PREDICTION
    # ==================================

    if st.button("Predict Tumor Class", type="primary"):

        with st.spinner("Analyzing MRI image..."):

            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(prediction[0])

            predicted_class = classes[predicted_index]

            confidence = float(
                prediction[0][predicted_index] * 100
            )

        with col2:
            st.subheader("Prediction Result")

            st.success(
                f"Predicted Class: {predicted_class.upper()}"
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.write("Class Scores:")

            for i, cls in enumerate(classes):
                st.write(
                    f"{cls}: {prediction[0][i] * 100:.2f}%"
                )

            st.progress(
                max(0.0, min(confidence / 100, 1.0))
            )

        st.divider()

        st.warning(
            "This is an experimental AI image classification "
            "result, not a medical diagnosis. It should not be "
            "used to determine whether someone has a brain tumor. "
            "Consult a qualified medical professional."
        )

else:
    st.info("Please upload an MRI image to start.")

# ==================================
# FOOTER
# ==================================

st.divider()

st.caption(
    "Brain Tumor Classification Project | "
    "Built with Python, TensorFlow and Streamlit"
)
