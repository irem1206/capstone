import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import urllib.request

ROBOFLOW_MODEL_ID = "alphabet-gesture-so0ya-1-rfdetr-small-t1"
ROBOFLOW_API_KEY = "irem-can/alphabet-gesture-so0ya-1-rfdetr-small-t1"

st.set_page_config(
    page_title="İşaret Dili Tanıma & Cümle Asistanı",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

st.markdown("""
    <div class="hero-container">
        <p class="hero-title">✋ İşaret Dili Tanıma & Cümle Kurma Asistanı</p>
        <p class="hero-subtitle">Yapay zeka tabanlı işaret dili tanıma sistemi ve interaktif kelime/cümle oluşturma paneli.</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            raw_labels = [line.strip() for line in f.readlines()]
        
        cleaned_labels = []
        for label in raw_labels:
            parts = label.split()
            if parts and parts[0].isdigit():
                parts = parts[1:]
            clean_name = " ".join(parts) if parts else label
            cleaned_labels.append(clean_name)
        return cleaned_labels
    except:
        return [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class_names = load_labels()

if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"
if "son_tahmin_harf" not in st.session_state:
    st.session_state.son_tahmin_harf = ""
if "son_tahmin_guven" not in st.session_state:
    st.session_state.son_tahmin_guven = 0.0

MODEL_YOLU = "model.tflite"
GITHUB_RAW_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"

@st.cache_resource(show_spinner=False)
def model_ve_etiketleri_yukle():
    if not os.path.exists(MODEL_YOLU):
        try:
            urllib.request.urlretrieve(GITHUB_RAW_URL, MODEL_YOLU)
        except Exception as e:
            return None, f"Model indirme hatası: {e}"
    
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_YOLU)
        interpreter.allocate_tensors()
        return interpreter, None
    except Exception as e:
        return None, f"Hata: {e}"

interpreter, model_hata = model_ve_etiketleri_yukle()

if model_hata:
    st.error(f"Sistem Başlatılamadı: {model_hata}")
else:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']

    def tahmin_uret(frame):
        expected_shape = input_details[0]['shape']
        target_size = (expected_shape[2], expected_shape[1])
        
        channels = expected_shape[3] if len(expected_shape) > 3 else 3
        
        if channels == 1:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        img_pil = Image.fromarray(processed_frame)
        img = ImageOps.fit(img_pil, target_size, Image.Resampling.LANCZOS)
        
        if input_dtype == np.float32:
            img_array = (np.asarray(img, dtype=np.float32) / 127.5) - 1.0
        else:
            img_array = np.asarray(img, dtype=input_dtype)
            
        if len(img_array.shape) == 2 and channels == 1:
            img_array = np.expand_dims(img_array, axis=-1)
            
        img_array = np.expand_dims(img_array, axis=0)
        
        safe_shape = [1 if d == -1 else d for d in expected_shape]
        img_array = np.reshape(img_array, safe_shape)

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

    st.header("🧠 Yapay Zeka Model Tahmin Paneli")
    input_mode = st.selectbox("Çalışma Modunu Seçin:", ("Görsel Yükleme (Test)", "Canlı Kamera Akışı (Gelişmiş)"))

    frame = None

    if input_mode == "Görsel Yükleme":
        uploaded_file = st.file_uploader("Modeli test etmek için bir işaret dili fotoğrafı seçin (.jpg, .png)...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Yüklenen Test Görüntüsü", width=300)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        st.warning("📹 Canlı kamera akışı için tarayıcı kamera izinlerinin açık olması gerekir. (WebRTC / OpenCV altyapısı)")
        camera_image = st.camera_input("Kameradan Anlık Görüntü Al")
        if camera_image is not None:
            bytes_data = camera_image.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            cam_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            st.image(cam_img, caption="Canlı Kare İşleniyor...", width=300)

    if frame is not None:
        try:
            class_name, confidence = tahmin_uret(frame)
            tahmin_harf = class_name.split()[0].upper() if " " in class_name else class_name.upper()
            st.session_state.son_tahmin_harf = tahmin_harf
            st.session_state.son_tahmin_guven = confidence
            
            st.success(f"🎯 Model Tahmini: **{tahmin_harf} Harfi** (Güven Skoru: %{confidence*100:.1f})")
            
            if st.button("➕ Bu Harfi Cümleye Ekle", type="primary"):
                st.session_state.biriken_metin += tahmin_harf
                st.rerun()
        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")

st.markdown("---")
st.header("🔤 İnteraktif Alfabe ve Cümle Paneli")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Harf Seçimi")
    harfler = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    cols = st.columns(6)
    for i, harf in enumerate(harfler):
        with cols[i % 6]:
            if st.button(harf, use_container_width=True, key=f"btn_grid_{harf}"):
                st.session_state.secilen_harf = harf
                st.session_state.biriken_metin += harf
                st.rerun()

    st.markdown("")
    if st.button("␣ Boşluk Ekle", use_container_width=True, type="secondary"):
        st.session_state.biriken_metin += " "
        st.rerun()

    st.markdown("")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Tümünü Temizle", use_container_width=True):
            st.session_state.biriken_metin = ""
            st.rerun()
    with col_b:
        if st.button("⬅️ Son Karakteri Sil", use_container_width=True):
            st.session_state.biriken_metin = st.session_state.biriken_metin[:-1]
            st.rerun()

with col2:
    st.subheader("📝 Metin Düzenleme Paneli")
    st.session_state.biriken_metin = st.text_input("Oluşan Cümle:", value=st.session_state.biriken_metin)
    
    metin_js = st.session_state.biriken_metin.replace("'", "\\'")
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
        margin-top: 5px;
        margin-bottom: 20px;
    ">🔊 Cümleyi Sesli Oku (Text-to-Speech)</button>
    """
    st.markdown(ses_butonu_html, unsafe_allow_html=True)
    
    current_letter = st.session_state.secilen_harf
    st.markdown(f"**Veri Seti Örnek Görseli ({current_letter}):**")
    
    dosya_adi = f"{current_letter}.png"
    if not os.path.exists(dosya_adi):
        dosya_adi = f"{current_letter}.jpg"
        
    if os.path.exists(dosya_adi):
        img_ornek = Image.open(dosya_adi)
        st.image(img_ornek, width=140, caption=f"Kaggle Veri Seti: {current_letter}")
    else:
        st.info(f"📌 '{current_letter}' için örnek görsel yüklenmemiş.")

st.markdown("---")
st.header("🤟 Cümlenin İşaret Dili Karşılığı")

metin = st.session_state.biriken_metin.upper()

if metin.strip():
    kelimeler = metin.split(' ')
    
    for kelime_idx, kelime in enumerate(kelimeler):
        if kelime:
            st.markdown(f"### 🔹 {kelime_idx + 1}. Kelime: **{kelime}**")
            
            harfler = [h for h in kelime if h.isalpha()]
            if harfler:
                cols = st.columns(min(len(harfler), 6))
                for idx, harf in enumerate(harfler):
                    col_index = idx % 6
                    with cols[col_index]:
                        resim_yolu = f"{harf}.png"
                        if not os.path.exists(resim_yolu):
                            resim_yolu = f"{harf}.jpg"
                        
                        if os.path.exists(resim_yolu):
                            img = Image.open(resim_yolu)
                            st.image(img, use_container_width=True, caption=f"{harf}")
                        else:
                            st.warning(f"'{harf}' yok")
            st.markdown("---")
else:
    st.info("💡 Yukarıdan harflere basarak veya model ile test ederek cümle oluşturun; kelimeler ve işaret dili görselleri burada gruplanarak listelensin.")
