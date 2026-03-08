import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
import tempfile
import google.generativeai as genai
from fertilizer_data import fertilizer_data

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Crop AI Assistant",
    page_icon="🌱",
    layout="centered"
)

# ------------------------------
# CUSTOM UI
# ------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.title{
text-align:center;
font-size:32px;
font-weight:bold;
color:white;
}

.card{
background:#1e293b;
padding:15px;
border-radius:10px;
margin-top:15px;
color:white;
}

body{
background-color:#0f172a;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌱 Smart Crop AI Assistant</div>", unsafe_allow_html=True)

st.write("Detect plant diseases and get fertilizer recommendations")

# ------------------------------
# LANGUAGE SELECTION
# ------------------------------

language = st.selectbox(
    "🌍 Select Language",
    ["English", "Hindi", "Telugu"]
)

# ------------------------------
# GEMINI API
# ------------------------------

if "GOOGLE_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    ai_model = genai.GenerativeModel("gemini-pro")
else:
    ai_model = None

# ------------------------------
# LOAD MODEL
# ------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5", compile=False)
    return model

model = load_model()

# ------------------------------
# CLASS NAMES (18 CLASSES)
# ------------------------------

class_names = [
"Apple___Apple_Scab",
"Apple___Healthy",
"Bell_Pepper___Bacterial_Spot",
"Bell_Pepper___Healthy",
"Cherry___Healthy",
"Cherry___Powdery_Mildew",
"Corn_Maize___Common_Rust_",
"Corn_Maize___Healthy",
"Grape___Black_Rot",
"Grape___Healthy",
"Peach___Bacterial_Spot",
"Peach___Healthy",
"Potato___Healthy",
"Potato___Late_Blight",
"Strawberry___Healthy",
"Strawberry___Leaf_Scorch",
"Tomato___Healthy",
"Tomato___Late_Blight"
]

# ------------------------------
# IMAGE PREPROCESS
# ------------------------------

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,axis=0)
    return img

# ------------------------------
# IMAGE INPUT
# ------------------------------

st.subheader("📷 Capture or Upload Crop Leaf")

camera = st.camera_input("Take Photo")
upload = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

image = None

if camera:
    image = Image.open(camera)

elif upload:
    image = Image.open(upload)

# ------------------------------
# PREDICTION
# ------------------------------

if image:

    st.image(image, caption="Leaf Image", use_column_width=True)

    img = preprocess(image)

    prediction = model.predict(img)

    pred = np.argmax(prediction)

    result = class_names[pred]

    crop = result.split("___")[0]
    disease = result.split("___")[1]

    st.markdown(f"""
    <div class="card">
    <h3>🌿 Crop : {crop}</h3>
    <h3>🦠 Disease : {disease}</h3>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# FERTILIZER DATA
# ------------------------------

    if result in fertilizer_data:

        fert = fertilizer_data[result]

        st.markdown(f"""
        <div class="card">
        <h3>🧪 Recommended Fertilizer</h3>
        <b>{fert['fertilizer_name']}</b>
        </div>
        """, unsafe_allow_html=True)

        if "image" in fert:
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

# ------------------------------
# VOICE ASSISTANT
# ------------------------------

        if st.button("🔊 Play Voice Guide"):

            lang_code = {"English":"en","Hindi":"hi","Telugu":"te"}[language]

            tts = gTTS(text=dosage, lang=lang_code)

            tmp = tempfile.NamedTemporaryFile(delete=False)

            tts.save(tmp.name)

            st.audio(tmp.name)

# ------------------------------
# AI CHAT ASSISTANT
# ------------------------------

st.markdown("---")

st.subheader("🧠 Ask Crop AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about crop diseases, fertilizers or farming...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    if ai_model:

        response = ai_model.generate_content(
            f"You are an agricultural expert helping farmers. Answer simply: {prompt}"
        )

        answer = response.text

    else:
        answer = "AI assistant not configured. Add GOOGLE_API_KEY in Streamlit secrets."

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})