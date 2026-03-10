import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
import tempfile
import zipfile
import os
from fertilizer_data import fertilizer_data

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Crop AI Assistant",
    page_icon="🌱",
    layout="centered"
)

# -------------------------
# STYLE
# -------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

body{
background-color:#0f172a;
}

.title{
text-align:center;
font-size:32px;
font-weight:bold;
color:white;
}

.card{
background:#1e293b;
padding:15px;
border-radius:12px;
margin-top:15px;
color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌱 Smart Crop AI Assistant</div>", unsafe_allow_html=True)

st.write("Detect crop diseases and get fertilizer recommendations")

# -------------------------
# UNZIP MODEL (FIRST RUN)
# -------------------------

if not os.path.exists("plant_disease_model.keras"):

    with zipfile.ZipFile("plant_disease_model.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# -------------------------
# LOAD MODEL
# -------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("plant_disease_model.keras")

    return model

with st.spinner("Loading AI model..."):
    model = load_model()

# -------------------------
# LANGUAGE
# -------------------------

language = st.selectbox(
    "🌍 Select Language",
    ["English","Hindi","Telugu"]
)

# -------------------------
# CLASS NAMES
# -------------------------

class_names = [

"Apple_Healthy",
"Apple_Scab",

"BellPepper_BacterialSpot",
"BellPepper_Healthy",

"Cherry_Healthy",
"Cherry_PowderyMildew",

"Corn_CommonRust",
"Corn_Healthy",

"Grape_BlackRot",
"Grape_Healthy",

"Peach_BacterialSpot",
"Peach_Healthy",

"Potato_Healthy",
"Potato_LateBlight",

"Strawberry_Healthy",
"Strawberry_LeafScorch",

"Tomato_Healthy",
"Tomato_LateBlight"
]

# -------------------------
# IMAGE PREPROCESS
# -------------------------

def preprocess(img):

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)

    return img

# -------------------------
# IMAGE UPLOAD
# -------------------------
st.subheader("📷 Capture or Upload Crop Leaf")

camera = st.camera_input("Take Photo")
upload = st.file_uploader("Upload Leaf Image", type=["jpg","jpeg","png"])

image = None

# If camera used
if camera is not None:
    image = Image.open(camera).convert("RGB")

# If uploaded
elif upload is not None:
    image = Image.open(upload).convert("RGB")

# -------------------------
# PREDICTION
# -------------------------

if image:

    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    img = preprocess(image)

    prediction = model.predict(img)

    pred = np.argmax(prediction)

    result = class_names[pred]

    confidence = np.max(prediction) * 100

    # Unknown detection
    if confidence < 70:

        st.warning("⚠ Unable to confidently identify the leaf. Please upload a clearer crop leaf image.")
        st.stop()

    crop, disease = result.split("_",1)

    st.markdown(f"""
    <div class="card">
    <h3>🌿 Crop : {crop}</h3>
    <h3>🦠 Disease : {disease}</h3>
    <h4>🎯 Confidence : {confidence:.2f}%</h4>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# FERTILIZER RECOMMENDATION
# -------------------------

    fert = fertilizer_data.get(result)

    if fert:

        st.markdown(f"""
        <div class="card">
        <h3>🧪 Recommended Fertilizer</h3>
        <b>{fert['fertilizer_name']}</b>
        </div>
        """, unsafe_allow_html=True)

        st.image(fert["image"], use_column_width=True)

        dosage = fert["dosage"][language]

        st.markdown(f"""
        <div class="card">
        <h4>📋 Dosage</h4>
        {dosage}
        </div>
        """, unsafe_allow_html=True)

        precaution = fert["precautions"][language]

        st.markdown(f"""
        <div class="card">
        <h4>⚠ Precautions</h4>
        {precaution}
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# VOICE GUIDE
# -------------------------

        if st.button("🔊 Play Voice Guide"):

            lang_code = {
                "English":"en",
                "Hindi":"hi",
                "Telugu":"te"
            }[language]

            tts = gTTS(text=dosage, lang=lang_code)

            tmp = tempfile.NamedTemporaryFile(delete=False)

            tts.save(tmp.name)

            st.audio(tmp.name)
