import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import os
import urllib.request
from gtts import gTTS
import base64

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="İşaret Dili Akıllı Çeviri ve Sesli Sentezleme",
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
        <p class="hero-title">🤟 Yapay Zeka Destekli Türk İşaret Dili ve Sesli İletişim Asistanı</p>
        <p class="hero-subtitle">Edge-AI Görsel Tanıma + gTTS Ses Sentezleme Motoru</p>
    </div>
""", unsafe_allow_html=True)

# --- MODEL VE KAYNAK YÖNETİMİ ---
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
            return None, [], "labels.txt dosyası bulunamadı."
    except Exception as e:
        return None, [], f"Hata: {e}"

interpreter, class_names, hata = model_ve_etiketleri_yukle()

if hata:
    st.error(hata)
else:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])

    def tahmin_uret(frame):
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = ImageOps.fit(img_pil, target_size, Image.Resampling.LANCZOS)
        
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(prediction[0])
        confidence = float(prediction[0][index])
        class_name = class_names[index] if index < len(class_names) else "Bilinmeyen"
        return class_name, confidence

    # --- ANA YERLEŞİM ---
    col_kamera, col_analiz = st.columns([1.2, 1], gap="large")

    with col_kamera:
        st.subheader("📹 Canlı Donanım Akışı")
        kamera_girdisi = st.camera_input("El işaretinizi gösterin ve kare yakalayın")

    with col_analiz:
        st.subheader("📊 Model Çıkarım & Doğruluk Filtresi")
        sonuc_alani = st.empty()
        guven_alani = st.empty()

    if kamera_girdisi is not None:
        bytes_data = kamera_girdisi.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        class_name, confidence = tahmin_uret(frame)
        harf_sade = class_name.split()[0].upper() if " " in class_name else class_name.upper()
        
        GUVEN_ESIGI = 0.70  
        
        if confidence < GUVEN_ESIGI:
            with sonuc_alani.container():
                st.markdown(f"""
                    <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #ef4444; text-align: center;'>
                        <h3 style='margin:0; color: #ef4444;'>⚠️ Düşük Güven Skoru</h3>
                        <h2 style='margin:10px 0; color: #f87171;'>Model Kararsız Kaldı</h2>
                        <p style='margin:0; color: #9ca3af;'>Tespit Edilen: {harf_sade} (Güven: %{confidence * 100:.1f})</p>
                    </div>
                """, unsafe_allow_html=True)
            harf_eklenebilir = False
        else:
            harf_eklenebilir = True
            with sonuc_alani.container():
                st.markdown(f"""
                    <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #10b981; text-align: center;'>
                        <span style='color: #9ca3af; font-size: 0.9em; text-transform: uppercase;'>Başarılı Tahmin</span>
                        <h1 style='margin: 10px 0; color: #34d399; font-size: 4em; font-weight: 800;'>{harf_sade}</h1>
                        <span style='color: #9ca3af; font-size: 0.85em;'>Sınıf Etiketi: {class_name}</span>
                    </div>
                """, unsafe_allow_html=True)
            
        with guven_alani.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model Olasılık Güven Skoru:**")
            st.progress(confidence)
            st.caption(f"Doğruluk Oranı: %{confidence * 100:.2f}")

        # --- CÜMLE VE SES SENTEZLEME BİRİMİ ---
        st.markdown("---")
        st.subheader("📝 Dinamik Cümle ve Ses Sentezleme Motoru")
        
        if 'cumle_hafizasi' not in st.session_state:
            st.session_state.cumle_hafizasi = ""
            
        col_islem1, col_islem2 = st.columns(2)
        with col_islem1:
            if st.button("➕ Harfi Cümleye Aktar", type="primary"):
                if harf_eklenebilir:
                    st.session_state.cumle_hafizasi += harf_sade
                else:
                    st.warning("Güven eşiğinin altında olduğu için eklenmedi!")
        with col_islem2:
            if st.button("␣ Boşluk Karakteri Ekle"):
                st.session_state.cumle_hafizasi += " "
                st.rerun()

        st.session_state.cumle_hafizasi = st.text_input("Oluşan Anlamlı Metin Çıktısı:", value=st.session_state.cumle_hafizasi)

        # --- YENİ ÖZELLİK: SESLİ OKUMA (TEXT-TO-SPEECH) ---
        if st.button("🔊 Cümleyi Sesli Oku (Text-to-Speech)"):
            if st.session_state.cumle_hafizasi.strip():
                try:
                    # gTTS ile Türkçe ses dosyası oluştur
                    tts = gTTS(text=st.session_state.cumle_hafizasi, lang='tr', slow=False)
                    ses_dosyasi = "cevirim_ses.mp3"
                    tts.save(ses_dosyasi)
                    
                    # Ses dosyasını HTML5 audio elementiyle çal
                    audio_file = open(ses_dosyasi, 'rb')
                    audio_bytes = audio_file.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode()
                    audio_html = f'<audio autoplay controls><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>'
                    st.markdown(audio_html, unsafe_allow_html=True)
                    st.success("Sesli sentezleme başarıyla gerçekleştirildi.")
                except Exception as e:
                    st.error(f"Ses sentezlenirken hata oluştu: {e}")
            else:
                st.warning("Okunacak metin boş!")

        col_temizle1, col_temizle2 = st.columns(2)
        with col_temizle1:
            if st.button("⬅️ Son Karakteri Sil"):
                st.session_state.cumle_hafizasi = st.session_state.cumle_hafizasi[:-1]
                st.rerun()
        with col_temizle2:
            if st.button("🧹 Belleği Sıfırla"):
                st.session_state.cumle_hafizasi = ""
                st.rerun()
    else:
        with col_analiz:
            st.info("👈 Analizi başlatmak için sol panelden kamerayı aktifleştirin.")

    # --- TEKNİK BİLGİ KARTI ---
    with st.expander("⚙️ Jüri & Sistem Mimarisi Detayları"):
        st.markdown(f"""
        - **Model Altyapısı:** TensorFlow Lite (`.tflite`) + gTTS (Google Text-to-Speech) Entegrasyonu
        - **Giriş Çözünürlüğü:** {target_size[0]}x{target_size[1]} piksel
        - **Sosyal Fayda / Vizyon:** İşaret dili kullanan bireylerin sesli iletişim kurabilmesini sağlayan akıllı sentezleme katmanı.
        """)
