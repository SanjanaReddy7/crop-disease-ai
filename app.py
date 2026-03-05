import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
from fertilizer_data import fertilizer_data
import tempfile

# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(page_title="Smart Crop AI", layout="centered")

st.title("🌱 Smart Crop Disease Assistant")

st.write("Detect crop diseases and get fertilizer recommendation")

# --------------------------------
# Language selection
# --------------------------------

language = st.selectbox("Select Language", ["English","Hindi","Telugu"])

# --------------------------------
# Load Model
# --------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("veg_model.h5")
    return model

model = load_model()

# --------------------------------
# Image preprocessing
# --------------------------------

def preprocess(img):

    img = img.resize((224,224))

    img = np.array(img)/255.0

    img = np.expand_dims(img, axis=0)

    return img

# --------------------------------
# Image Input Option
# --------------------------------

st.subheader("📷 Capture or Upload Crop Leaf")

camera_image = st.camera_input("Take a picture")

upload_image = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

image = None

if camera_image:
    image = Image.open(camera_image)

elif upload_image:
    image = Image.open(upload_image)

# --------------------------------
# Prediction
# --------------------------------

if image:

    st.image(image, caption="Selected Leaf Image", use_column_width=True)

    img = preprocess(image)

    prediction = model.predict(img)

    pred_class = np.argmax(prediction)

    if pred_class == 0:
        result = "Diseased"
    else:
        result = "Healthy"

    st.markdown(f"### 🌿 Prediction: {result}")

# --------------------------------
# Fertilizer Recommendation
# --------------------------------

    fert = fertilizer_data[result]

    st.markdown(f"### 🧪 Recommended Fertilizer: {fert['fertilizer_name']}")

    st.image(fert["image"], caption=fert["fertilizer_name"], width=250)

    dosage = fert["dosage"][language]

    st.markdown("### 📋 Dosage Instructions")

    st.write(dosage)

# --------------------------------
# Voice Assistant
# --------------------------------

    if st.button("🔊 Play Voice Guide"):

        lang_code = {"English":"en","Hindi":"hi","Telugu":"te"}[language]

        tts = gTTS(text=dosage, lang=lang_code)

        tmp = tempfile.NamedTemporaryFile(delete=False)

        tts.save(tmp.name)

        st.audio(tmp.name)

# --------------------------------
# AI Chat Assistant
# --------------------------------

st.markdown("---")

st.subheader("🧠 Crop AI Assistant")

question = st.text_input("Ask about crop diseases or fertilizers")

if question:

    response = "Maintain proper irrigation, monitor leaf spots regularly, and apply recommended fertilizers."

    st.write("👨‍🌾 You:", question)

    st.write("🤖 Assistant:", response)