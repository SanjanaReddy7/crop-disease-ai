import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
from fertilizer_data import fertilizer_data
import tempfile
import google.generativeai as genai

# ------------------------------
# PAGE CONFIG
# ------------------------------

st.set_page_config(
    page_title="Crop AI Assistant",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------------------
# HIDE STREAMLIT HEADER
# ------------------------------

st.markdown("""
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

body{
background-color:#0f172a;
}

.card{
background:#1e293b;
padding:15px;
border-radius:10px;
margin-top:10px;
color:white;
}

.title{
text-align:center;
font-size:28px;
font-weight:bold;
color:white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# TITLE
# ------------------------------

st.markdown("<div class='title'>🌱 Smart Crop AI Assistant</div>", unsafe_allow_html=True)

st.write("Detect crop diseases and get fertilizer recommendations")

# ------------------------------
# LANGUAGE SELECTION
# ------------------------------

language = st.selectbox(
    "🌍 Select Language",
    ["English","Hindi","Telugu"]
)

# ------------------------------
# GOOGLE GEMINI API
# ------------------------------



import google.generativeai as genai

# Configure API
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Load Gemini model
ai_model = genai.GenerativeModel("gemini-1.5-flash-latest")

# ------------------------------
# LOAD CNN MODEL
# ------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("veg_model.h5")

    return model

model = load_model()

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

image=None

if camera:
    image=Image.open(camera)

elif upload:
    image=Image.open(upload)

# ------------------------------
# PREDICTION
# ------------------------------

if image:

    st.image(image,caption="Leaf Image",use_column_width=True)

    img=preprocess(image)

    prediction=model.predict(img)

    pred=np.argmax(prediction)

    if pred==0:
        result="Diseased"
    else:
        result="Healthy"

    st.markdown(f"### 🌿 Prediction: {result}")

# ------------------------------
# FERTILIZER RECOMMENDATION
# ------------------------------

    fert=fertilizer_data[result]

    st.markdown(f"""
    <div class="card">
    <h3>🧪 Recommended Fertilizer</h3>
    <b>{fert['fertilizer_name']}</b>
    </div>
    """,unsafe_allow_html=True)

    st.image(fert["image"],use_column_width=True)

    dosage=fert["dosage"][language]

    st.markdown(f"""
    <div class="card">
    <h4>📋 Instructions</h4>
    {dosage}
    </div>
    """,unsafe_allow_html=True)

# ------------------------------
# VOICE ASSISTANT
# ------------------------------

    if st.button("🔊 Play Voice Guide"):

        lang_code={"English":"en","Hindi":"hi","Telugu":"te"}[language]

        tts=gTTS(text=dosage,lang=lang_code)

        tmp=tempfile.NamedTemporaryFile(delete=False)

        tts.save(tmp.name)

        st.audio(tmp.name)

# ------------------------------
# AI CHAT ASSISTANT
# ------------------------------

st.markdown("---")

st.subheader("🧠 Ask Crop AI Assistant")

if "messages" not in st.session_state:
    st.session_state.messages=[]

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

prompt=st.chat_input("Ask about crop diseases, fertilizers or farming...")

if prompt:

    st.session_state.messages.append({"role":"user","content":prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    response = ai_model.generate_content(
    f"You are an agricultural expert helping farmers. Answer simply: {prompt}"
)

answer = response.text

    answer=response.text

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant","content":answer})
