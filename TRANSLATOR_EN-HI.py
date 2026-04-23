import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="English Hindi Translator 📚",
    page_icon="📖",
    layout="centered"
)

# -------------------------------
# Session State
# -------------------------------
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# -------------------------------
# STYLING
# -------------------------------
st.markdown("""
<style>

/* Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1519389950473-47ba0277781c");
    background-size: cover;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
}

/* Overlay */
.stApp::before {
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: rgba(250, 245, 255, 0.75);
    z-index: 0;
}

/* Hide header */
header {visibility: hidden;}

.block-container {
    position: relative;
    z-index: 1;
}

/* MAIN HEADING */
.header-box {
    background: #d1c4e9;
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    font-style: italic;
    text-decoration: underline;
    color: #311b92;
}

/* SUBHEADING */
.sub-box {
    background: #e6dcf5;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    margin: 15px 0 25px 0;
}

/* SMALL HEADINGS */
.heading-box {
    background: #e6dcf5;
    padding: 8px;
    border-radius: 8px;
    font-weight: bold;
    color: #311b92;
}

/* MAIN CARD */
.main-box {
    background: rgba(255,255,255,0.95);
    padding: 30px;
    border-radius: 20px;
}

/* TEXTAREA */
textarea {
    border-radius: 10px;
}

/* BUTTONS */
.stButton > button {
    background: linear-gradient(90deg, #7e57c2, #5e35b1);
    color: white;
    border-radius: 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODEL (BEST)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# UI START
# -------------------------------
st.markdown('<div class="main-box">', unsafe_allow_html=True)

# ✅ ATTRACTIVE HEADING
st.markdown(
    '<div class="header-box">📚✨ 🤖 <u><i>English → Hindi Translator</i></u> ✏️📖</div>',
    unsafe_allow_html=True
)

# Subheading
st.markdown(
    '<div class="sub-box">Translate any English text into Hindi easily ✨</div>',
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Settings")

max_length = st.sidebar.slider("Output Length", 20, 200, 100)
text_height = st.sidebar.slider("Text Box Height", 100, 400, 150)
text_size = st.sidebar.slider("Text Size", 14, 30, 18)

output_color = st.sidebar.color_picker("Hindi Text Color", "#311b92")
input_color = st.sidebar.color_picker("Input Text Color", "#000000")

# -------------------------------
# INPUT
# -------------------------------
st.markdown('<div class="heading-box">✍️ Enter English Text:</div>', unsafe_allow_html=True)

# Apply input style
st.markdown(f"""
<style>
textarea {{
    color:{input_color}!important;
    font-size:{text_size}px!important;
}}
</style>
""", unsafe_allow_html=True)

input_text = st.text_area("", height=text_height, key="input_text")

# -------------------------------
# TRANSLATION FUNCTION
# -------------------------------
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt")

    translated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("hin_Deva"),
        max_length=max_length
    )

    return tokenizer.decode(translated[0], skip_special_tokens=True)

# -------------------------------
# BUTTONS
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Translate"):
        if not st.session_state.input_text.strip():
            st.warning("⚠️ Please enter text")
        else:
            with st.spinner("Translating..."):
                result = translate_text(st.session_state.input_text)

                st.markdown('<div class="heading-box">🇮🇳 Hindi Translation:</div>', unsafe_allow_html=True)

                st.markdown(
                    f"<div style='background:#e6dcf5;padding:15px;border-radius:10px;'>"
                    f"<h3 style='color:{output_color};'>👉 {result}</h3></div>",
                    unsafe_allow_html=True
                )

with col2:
    if st.button("🧹 Clear"):
        st.session_state.input_text = ""
        st.rerun()

# -------------------------------
# HOW IT WORKS (ADDED ✅)
# -------------------------------
st.markdown("---")
st.markdown('<div class="heading-box">🧠 How it works</div>', unsafe_allow_html=True)

st.write("""
✍️ Enter English text  
📚 AI translates into Hindi  
🎨 Customize colors  
⚡ Get instant result  

Comfortable & student-friendly 😄
""")

# -------------------------------
# END
# -------------------------------
st.markdown('</div>', unsafe_allow_html=True)