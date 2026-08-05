import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import urllib.request

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="İşaret Dili Akıllı Çeviri ve Sentezleme Sistemi",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN KURUMSAL STİLLER ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .hero-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #374151;
        text-align: center;
        margin-bottom: 25px;
    }
    .hero-title { color: #60a5fa; font-size: 2.2em; font-weight: 800; margin: 0; }
    .hero-subtitle { color: #9ca3af; font-size: 1.1em; margin-top: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2563eb; color: white; border: none; }
    .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

# --- ÜST KISIM (HERO SECTION) ---
st.markdown("""
    <div class="hero-container">
        <p class="hero-title">🤟 Yapay Zeka Destekli Türk İşaret Dili Çeviri Asistanı</p>
        <p class="hero-subtitle">Edge-AI Tabanlı Gerçek Zamanlı Görsel Tanıma ve Dinamik Cümle Sentezleme Motoru</p>
    </div>
""", unsafe_allow_html=True)

# --- OTURUM HAFIZASI (STATE) ---
if 'cumle_hafizasi' not in st.session_state:
    st.session_state.cumle_hafizasi = ""
if 'aktif_harf' not in st.session_state:
    st.session_state.aktif_harf = "A"
if 'aktif_guven' not in st.session_state:
    st.session_state.aktif_guven = 0.95

# --- MODEL VE ETİKET YÖNETİMİ ---
MODEL_YOLU = "model.tflite"
ETIKET_YOLU = "labels.txt"
GITHUB_RAW_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"

@st.cache_resource(show_spinner=False)
def model_ve_etiketleri_yukle():
    if not os.path.exists(MODEL_YOLU):
        try:
            urllib.request.urlretrieve(GITHUB_RAW_URL, MODEL_YOLU)
        except Exception as e:
            return None, [], f"Model indirme hatası: {e}"
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_YOLU)
        interpreter.allocate_tensors()
        
        if os.path.exists(ETIKET_YOLU):
            with open(ETIKET_YOLU, "r", encoding="utf-8") as f:
                raw_labels = [line.strip() for line in f.readlines()]
            
            class_names = []
            for label in raw_labels:
                parts = label.split()
                if parts and parts[0].isdigit():
                    parts = parts[1:]
                clean_name = " ".join(parts) if parts else label
                class_names.append(clean_name)
            return interpreter, class_names, None
        else:
            default_labels = ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V", "Y", "Z"]
            return interpreter, default_labels, None
    except Exception as e:
        return None, [], f"Hata: {e}"

interpreter, class_names, hata = model_ve_etiketleri_yukle()

if hata:
    st.error(hata)
else:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    input_dtype = input_details[0]['dtype']

    def tahmin_uret(frame):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = ImageOps.fit(img_pil, target_size, Image.Resampling.LANCZOS)
        
        if input_dtype == np.float32:
            img_array = np.asarray(img, dtype=np.float32) / 255.0
        else:
            img_array = np.asarray(img, dtype=input_dtype)
            
        img_array = np.expand_dims(img_array, axis=0)
        
        # Boyut uyuşmazlığı hatalarını önleyen güvenli reshape
        expected_shape = [1 if d == -1 else d for d in input_details[0]['shape']]
        img_array = np.reshape(img_array, expected_shape)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        pred_vals = prediction[0]
        index = np.argmax(pred_vals)
        confidence = float(pred_vals[index])
        if confidence > 1.0:
            confidence = confidence / 255.0

        class_name = class_names[index % len(class_names)] if len(class_names) > 0 else "Bilinmeyen"
        return class_name, confidence

    # --- 1. BÖLÜM: GİRDİ YÖNTEMİ SEÇİMİ (FOTOĞRAF YÜKLE / KAMERA) ---
    col_kamera, col_analiz = st.columns([1.2, 1], gap="large")

    with col_kamera:
        st.subheader("📹 Canlı Veri Akışı ve Test")
        input_mode = st.selectbox("Çalışma Modunu Seçin:", ("Görsel Yükleme (Test)", "Canlı Kamera Akışı"))

        frame = None
        if input_mode == "Görsel Yükleme (Test)":
            yuklenen_dosya = st.file_uploader("Modeli test etmek için bir işaret dili fotoğrafı seçin (.jpg, .png)...", type=["jpg", "jpeg", "png"])
            if yuklenen_dosya is not None:
                image = Image.open(yuklenen_dosya).convert('RGB')
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            st.warning("📹 Canlı kamera akışı için tarayıcı kamera izinlerinin açık olması gerekir.")
            kamera_girdisi = st.camera_input("Kameradan Anlık Görüntü Al")
            if kamera_girdisi is not None:
                bytes_data = kamera_girdisi.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with col_analiz:
        st.subheader("📊 Model Çıkarım Paneli")
        sonuc_alani = st.empty()
        guven_alani = st.empty()

    if frame is not None:
        try:
            class_name, confidence = tahmin_uret(frame)
            st.session_state.aktif_harf = class_name.split()[0].upper() if " " in class_name else class_name.upper()
            st.session_state.aktif_guven = confidence
        except Exception:
            pass

    harf_sade = st.session_state.aktif_harf
    confidence = st.session_state.aktif_guven
    GUVEN_ESIGI = 0.70  

    if confidence < GUVEN_ESIGI and frame is not None:
        with sonuc_alani.container():
            st.markdown(f"""
                <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #ef4444; text-align: center;'>
                    <h3 style='margin:0; color: #ef4444;'>⚠️ Düşük Güven Skoru</h3>
                    <h2 style='margin:10px 0; color: #f87171;'>Model Kararsız Kaldı</h2>
                    <p style='margin:0; color: #9ca3af;'>Tespit Edilen: {harf_sade} (Güven: %{confidence * 100:.1f})</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        with sonuc_alani.container():
            st.markdown(f"""
                <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #10b981; text-align: center;'>
                    <span style='color: #9ca3af; font-size: 0.9em; text-transform: uppercase;'>Aktif Tahmin / Seçim</span>
                    <h1 style='margin: 10px 0; color: #34d399; font-size: 4em; font-weight: 800;'>{harf_sade}</h1>
                    <span style='color: #9ca3af; font-size: 0.85em;'>Güven Skoru: %{confidence * 100:.1f}</span>
                </div>
            """, unsafe_allow_html=True)
        
    with guven_alani.container():
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Model Olasılık Güven Skoru:**")
        st.progress(confidence)
        st.caption(f"Doğruluk Oranı: %{confidence * 100:.2f}")

# --- 2. BÖLÜM: DİNAMİK CÜMLE VE SES SENTEZLEME MOTORU ---
st.markdown("---")
st.subheader("📝 Metin ve Cümle Sentezleme Paneli")

col_islem1, col_islem2 = st.columns(2)
with col_islem1:
    if st.button("➕ Aktif Harfi Cümleye Ekle", type="primary"):
        st.session_state.cumle_hafizasi += st.session_state.aktif_harf
        st.rerun()
with col_islem2:
    if st.button("␣ Boşluk Karakteri Ekle"):
        st.session_state.cumle_hafizasi += " "
        st.rerun()

st.session_state.cumle_hafizasi = st.text_input("Oluşan Anlamlı Metin Çıktısı:", value=st.session_state.cumle_hafizasi)

# --- TARAYICI TABANLI SESLİ OKUMA (WEB SPEECH API) ---
metin_js = st.session_state.cumle_hafizasi.replace("'", "\\'")
ses_butonu_html = f"""
<button onclick="
    const utterance = new SpeechSynthesisUtterance('{metin_js}');
    utterance.lang = 'tr-TR';
    window.speechSynthesis.speak(utterance);
" style="
    width: 100%;
    background-color: #2563eb;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    margin-top: 10px;
">🔊 Cümleyi Sesli Oku (Text-to-Speech)</button>
"""
st.markdown(ses_butonu_html, unsafe_allow_html=True)

col_temizle1, col_temizle2 = st.columns(2)
with col_temizle1:
    if st.button("⬅️ Son Karakteri Sil"):
        st.session_state.cumle_hafizasi = st.session_state.cumle_hafizasi[:-1]
        st.rerun()
with col_temizle2:
    if st.button("🧹 Belleği Temizle"):
        st.session_state.cumle_hafizasi = ""
        st.rerun()

# --- 3. BÖLÜM: SANAL KLAVYE (HER ZAMAN GÖRÜNÜR) ---
st.markdown("---")
st.subheader("⌨️ İnteraktif Harf Matrisi (Sanal Klavye)")
st.markdown("<p style='color: #9ca3af; font-size: 0.9em;'>Harflere tıkladığınızda yukarıdaki model çıktısı paneli anında o harfe güncellenir ve kelimeye eklenir:</p>", unsafe_allow_html=True)

alfabe_satirlari = [
    ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ"],
    ["H", "I", "İ", "J", "K", "L", "M", "N", "O"],
    ["Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V"],
    ["Y", "Z"]
]

for satir in alfabe_satirlari:
    cols = st.columns(len(satir))
    for i, harf in enumerate(satir):
        with cols[i]:
            if st.button(harf, key=f"klavye_{harf}", use_container_width=True):
                st.session_state.aktif_harf = harf
                st.session_state.aktif_guven = 1.0  
                st.session_state.cumle_hafizasi += harf
                st.rerun()
