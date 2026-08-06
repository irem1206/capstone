import streamlit as st
import numpy as np
import cv2
import requests
import os
import base64
from PIL import Image

st.set_page_config(
    page_title="İşaret Dili Tanıma",
    page_icon="🤟",
    layout="wide"
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

ROBOFLOW_MODEL_ID = "alphabet-gesture-so0ya/1"
ROBOFLOW_API_KEY = "ggw75nomaYTUJtoijwI4"

if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"
if "son_tahmin_harf" not in st.session_state:
    st.session_state.son_tahmin_harf = ""
if "son_tahmin_guven" not in st.session_state:
    st.session_state.son_tahmin_guven = 0.0

def roboflow_tahmin_yap(frame):
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "BURAYA_API_KEYINIZI_YAZIN":
        st.error("🔑 Lütfen ROBOFLOW_API_KEY alanını doldurun.")
        return frame, "API KEY EKSİK", 0.0

    if not ROBOFLOW_MODEL_ID:
        st.error("🏷️ Lütfen ROBOFLOW_MODEL_ID alanını doldurun.")
        return frame, "MODEL ID EKSİK", 0.0

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', frame_rgb)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        upload_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}"
        params = {
            "api_key": ROBOFLOW_API_KEY,
            "confidence": 15
        }
        
        response = requests.post(
            upload_url,
            params=params,
            data=img_base64,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code != 200:
            st.error(f"Roboflow API Hatası ({response.status_code}): {response.text}")
            return frame_rgb, "API Hatası", 0.0

        predictions = response.json().get("predictions", [])
        
        if not predictions:
            return frame_rgb, "Bilinmeyen", 0.0

        processed_frame = frame_rgb.copy()
        best_class = "Bilinmeyen"
        max_conf = 0.0

        for pred in predictions:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            label = str(pred['class']).upper()
            conf = float(pred['confidence'])

            if conf > max_conf:
                max_conf = conf
                best_class = label

            pt1 = (int(x - w / 2), int(y - h / 2))
            pt2 = (int(x + w / 2), int(y + h / 2))

            cv2.rectangle(processed_frame, pt1, pt2, (0, 255, 0), 3)
            cv2.putText(
                processed_frame, 
                f"{label} %{int(conf*100)}", 
                (pt1[0], max(pt1[1] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )

        return processed_frame, best_class, max_conf

    except Exception as e:
        st.error(f"Kod içi hata: {e}")
        return frame, "Hata", 0.0

st.header("🧠 Yapay Zeka Model Tahmin Paneli")
input_mode = st.selectbox("Çalışma Modunu Seçin:", ("Görsel Yükleme (Test)", "Canlı Kamera Akışı (Gelişmiş)"))

frame = None

if input_mode == "Görsel Yükleme (Test)":
    uploaded_file = st.file_uploader("Modeli test etmek için bir fotoğraf seçin...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
else:
    st.warning("📹 Canlı kamera akışı için tarayıcı kamera izinlerinin açık olması gerekir.")
    camera_image = st.camera_input("Kameradan Anlık Görüntü Al")
    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

if frame is not None:
    try:
        processed_img, tahmin_harf, confidence = roboflow_tahmin_yap(frame)
        st.session_state.son_tahmin_harf = tahmin_harf
        st.session_state.son_tahmin_guven = confidence
        
        st.image(processed_img, caption="Roboflow Nesne Tespiti Sonucu", width=350)
        
        if confidence > 0.15:
            st.success(f"🎯 Model Tahmini: **{tahmin_harf} Harfi** (Güven Skoru: %{confidence*100:.1f})")
            if st.button("➕ Bu Harfi Cümleye Ekle", type="primary"):
                st.session_state.biriken_metin += tahmin_harf
                st.rerun()
        else:
            st.warning("Ekranda belirgin bir el işareti algılanamadı.")
            
    except Exception as e:
        st.error(f"Tahmin sırasında hata oluştu: {e}")

st.markdown("---")
st.header("🔤 İnteraktif Alfabe ve Cümle Paneli")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Harf Seçim Matrisi")
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
    ses_butonu_html = f"""<button onclick="const utterance = new SpeechSynthesisUtterance('{metin_js}'); utterance.lang = 'tr-TR'; window.speechSynthesis.speak(utterance);" style="width: 100%; background-color: #2563eb; color: white; padding: 10px 20px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 5px; margin-bottom: 20px;">🔊 Cümleyi Sesli Oku (Text-to-Speech)</button>"""
    st.markdown(ses_butonu_html, unsafe_allow_html=True)

st.markdown("---")
st.header("🤟 Cümlenin İşaret Dili Karşılığı (Görsel Akış)")

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
    st.info("💡 Yukarıdan harflere basarak veya model ile test ederek cümle oluşturun.")
