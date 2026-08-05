import streamlit as st
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image, ImageOps

# --- TFLITE YÜKLEME ---
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        Interpreter = None

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="İşaret Dili Asistanı", page_icon="🤟", layout="wide", initial_sidebar_state="collapsed")

# --- MODERN STİLLER ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .hero-container { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 30px; border-radius: 15px; border: 1px solid #374151; text-align: center; margin-bottom: 25px; }
    .hero-title { color: #60a5fa; font-size: 2.2em; font-weight: 800; margin: 0; }
    .hero-subtitle { color: #9ca3af; font-size: 1.1em; margin-top: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2563eb; color: white; border: none; }
    .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='hero-container'>
        <p class='hero-title'>🤟 Yapay Zeka Destekli Türk İşaret Dili Çeviri Asistanı</p>
        <p class='hero-subtitle'>Edge-AI Tabanlı Gerçek Zamanlı Görsel Tanıma ve Dinamik Cümle Sentezleme Motoru</p>
    </div>
""", unsafe_allow_html=True)

# --- HAFIZA ---
if 'cumle' not in st.session_state: st.session_state.cumle = ""
if 'aktif_harf' not in st.session_state: st.session_state.aktif_harf = ""
if 'aktif_guven' not in st.session_state: st.session_state.aktif_guven = 0.0

# --- MODEL YÜKLEME ---
MODEL_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"
if not os.path.exists("model.tflite"):
    try: urllib.request.urlretrieve(MODEL_URL, "model.tflite")
    except: pass

@st.cache_resource(show_spinner=False)
def modeli_hazirla():
    if not Interpreter or not os.path.exists("model.tflite"): return None, []
    try:
        interp = Interpreter(model_path="model.tflite")
        interp.allocate_tensors()
        labels = ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"]
        if os.path.exists("labels.txt"):
            with open("labels.txt", "r", encoding="utf-8") as f:
                labels = [line.strip().split(" ", 1)[-1] if line.strip()[0].isdigit() else line.strip() for line in f.readlines()]
        return interp, labels
    except:
        return None, []

interpreter, class_names = modeli_hazirla()

def tahmin_yap(frame):
    if not interpreter: return "Bilinmeyen", 0.0
    try:
        in_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]
        
        img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (in_det['shape'][2], in_det['shape'][1]))
        
        if in_det['dtype'] == np.float32: img_arr = np.float32(img) / 255.0
        else: img_arr = img.astype(in_det['dtype'])
            
        img_arr = np.expand_dims(img_arr, axis=0)
        
        expected_shape = [1 if d == -1 else d for d in in_det['shape']]
        img_arr = np.reshape(img_arr, expected_shape)

        interpreter.set_tensor(in_det['index'], img_arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(out_det['index'])[0]
        
        idx = np.argmax(preds)
        conf = float(preds[idx]) / 255.0 if float(preds[idx]) > 1.0 else float(preds[idx])
        return class_names[idx] if idx < len(class_names) else "Bilinmeyen", conf
    except Exception:
        return "Hata", 0.0

# --- ARAYÜZ ---
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("📹 Canlı Veri Akışı")
    girdi = st.radio("Seçim:", ["Kamera", "Fotoğraf"], horizontal=True, label_visibility="collapsed")
    frame = None
    if girdi == "Fotoğraf":
        file = st.file_uploader("Resim seç", type=["jpg", "png"])
        if file:
            frame = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        cam = st.camera_input("Kamera")
        if cam:
            frame = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)

if frame is not None:
    h, c = tahmin_yap(frame)
    st.session_state.aktif_harf = h.split()[0].upper() if " " in h else h.upper()
    st.session_state.aktif_guven = c

with col2:
    st.subheader("📊 Model Çıkarım Paneli")
    if st.session_state.aktif_harf == "":
        st.info("Kameradan, fotoğraftan veya en aşağıdaki klavyeden bir giriş bekliyorum...")
    else:
        harf = st.session_state.aktif_harf
        guven = st.session_state.aktif_guven
        
        if guven < 0.70 and frame is not None:
            st.markdown(f"<div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #ef4444; text-align: center;'><h3 style='margin:0; color: #ef4444;'>⚠️ Düşük Güven</h3><h1 style='margin:10px 0; color: #f87171;'>{harf}</h1><p style='color: #9ca3af;'>Güven: %{guven*100:.1f}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #10b981; text-align: center;'><span style='color: #9ca3af; text-transform: uppercase;'>Aktif Tahmin / Seçim</span><h1 style='margin: 10px 0; color: #34d399; font-size: 5em;'>{harf}</h1><span style='color: #9ca3af;'>Güven Skoru: %{guven*100:.1f}</span></div>", unsafe_allow_html=True)
        st.progress(guven)

# --- METİN VE SES SENTEZLEME ---
st.markdown("---")
st.subheader("📝 Metin ve Ses Motoru")

c1, c2 = st.columns(2)
if c1.button("➕ Yukarıdaki Harfi Ekle", type="primary"):
    if st.session_state.aktif_harf:
        st.session_state.cumle += st.session_state.aktif_harf
        st.rerun()
if c2.button("␣ Boşluk Ekle"):
    st.session_state.cumle += " "
    st.rerun()

st.session_state.cumle = st.text_input("Çıktı:", value=st.session_state.cumle, label_visibility="collapsed")

metin_js = st.session_state.cumle.replace("'", "\\'")
st.markdown(f"<button onclick=\"const u = new SpeechSynthesisUtterance('{metin_js}'); u.lang='tr-TR'; window.speechSynthesis.speak(u);\" style='width: 100%; background-color: #2563eb; color: white; padding: 10px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer;'>🔊 Sesli Oku</button>", unsafe_allow_html=True)

c3, c4 = st.columns(2)
if c3.button("⬅️ Sil"):
    st.session_state.cumle = st.session_state.cumle[:-1]
    st.rerun()
if c4.button("🧹 Temizle"):
    st.session_state.cumle = ""
    st.rerun()

# --- EN ALT KISIM: SANAL KLAVYE ---
st.markdown("---")
st.subheader("⌨️ Sanal Klavye")

alfabe = [
    ["A","B","C","Ç","D","E","F","G","Ğ"], 
    ["H","I","İ","J","K","L","M","N","O"], 
    ["Ö","P","R","S","Ş","T","U","Ü","V"], 
    ["Y","Z"]
]

for satir in alfabe:
    cols = st.columns(len(satir))
    for i, harf in enumerate(satir):
        if cols[i].button(harf, key=f"k_{harf}"):
            st.session_state.aktif_harf = harf
            st.session_state.aktif_guven = 1.0  
            st.session_state.cumle += harf
            st.rerun()
