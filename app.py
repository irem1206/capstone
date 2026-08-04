import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import urllib.request

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="İşaret Dili Gerçek Zamanlı Çeviri Sistemi",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- KURUMSAL VE MODERN ARAYÜZ STİLLERİ ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2b313e; color: white; border: 1px solid #414855; }
    .stButton>button:hover { background-color: #ff4b4b; border-color: #ff4b4b; }
    .metric-box { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# --- BAŞLIK VE PROJE KÜNYESİ ---
st.title("🧠 Edge-AI Destekli İşaret Dili Tanıma ve Cümle Sentezleme Asistanı")
st.markdown("<p style='color: #8b949e;'>Teachable Machine TFLite Motoru ile Gerçek Zamanlı Görsel Çıkarım Sistemi</p>", unsafe_allow_html=True)
st.markdown("---")

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
        img = img_pil.resize(target_size)
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        index = np.argmax(prediction[0])
        confidence = float(prediction[0][index])
        class_name = class_names[index] if index < len(class_names) else "Bilinmeyen"
        return class_name, confidence

    # --- ANA YERLEŞİM (LAYOUT) ---
    col_kamera, col_analiz = st.columns([1.3, 1], gap="large")

    with col_kamera:
        st.subheader("📹 Canlı Donanım Sensör Akışı")
        # Fotoğraf yükleme seçenekleri tamamen kaldırıldı. Sadece kamera aktif.
        kamera_girdisi = st.camera_input("Model için anlık kare yakalayın")

    with col_analiz:
        st.subheader("📊 Derin Öğrenme Çıkarım Paneli")
        sonuc_alani = st.empty()
        guven_alani = st.empty()

    if kamera_girdisi is not None:
        bytes_data = kamera_girdisi.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        class_name, confidence = tahmin_uret(frame)
        harf_sade = class_name.split()[0].upper() if " " in class_name else class_name.upper()
        
        # Profesyonel Kart Görünümü
        with sonuc_alani.container():
            st.markdown(f"""
                <div style='background-color: #161b22; padding: 25px; border-radius: 12px; border: 1px solid #30363d; text-align: center;'>
                    <span style='color: #8b949e; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;'>Tahmin Edilen Sınıf</span>
                    <h1 style='margin: 10px 0; color: #58a6ff; font-size: 4em; font-weight: 800;'>{harf_sade}</h1>
                    <span style='color: #8b949e; font-size: 0.85em;'>Model Etiketi: {class_name}</span>
                </div>
            """, unsafe_allow_html=True)
            
        with guven_alani.container():
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Model Olasılık Dağılımı (Confidence):**")
            st.progress(confidence)
            st.caption(f"Doğruluk Güven Skoru: %{confidence * 100:.2f}")

        # --- CÜMLE VE METİN SENTEZLEME BİRİMİ ---
        st.markdown("---")
        st.subheader("📝 Akıllı Cümle Sentezleme Motoru")
        
        if 'cumle_hafizasi' not in st.session_state:
            st.session_state.cumle_hafizasi = ""
            
        col_islem1, col_islem2 = st.columns(2)
        with col_islem1:
            if st.button("➕ Tespit Edilen Harfi Ekle", type="primary"):
                st.session_state.cumle_hafizasi += harf_sade
        with col_islem2:
            if st.button("␣ Boşluk Karakteri Ekle"):
                st.session_state.cumle_hafizasi += " "
                st.rerun()

        st.session_state.cumle_hafizasi = st.text_input("Oluşan Anlamlı Metin Çıktısı:", value=st.session_state.cumle_hafizasi)

        col_temizle1, col_temizle2 = st.columns(2)
        with col_temizle1:
            if st.button("⬅️ Son Karakteri Geri Al"):
                st.session_state.cumle_hafizasi = st.session_state.cumle_hafizasi[:-1]
                st.rerun()
        with col_temizle2:
            if st.button("🧹 Belleği Sıfırla"):
                st.session_state.cumle_hafizasi = ""
                st.rerun()
    else:
        with col_analiz:
            st.info("👈 Analizi başlatmak ve gerçek zamanlı çıkarım almak için sol panelden kamerayı aktifleştirin.")

    # --- JÜRİ İÇİN TEKNİK DOKÜMANTASYON KARTI ---
    with st.expander("⚙️ Sistem Mimarisi ve Teknik Detaylar (Jüri Bilgi Kartı)"):
        st.markdown(f"""
        - **Model Altyapısı:** TensorFlow Lite (`.tflite`) optimize edilmiş hafif sinir ağı.
        - **Giriş Boyutlandırma:** {target_size[0]}x{target_size[1]} piksel normalize edilmiş tensör matrisi.
        - **Performans Optimizasyonu:** `st.cache_resource` dekoratörü ile bellek katmanında önbellekleme.
        - **Donanım Uyumluluğu:** Edge cihazlar ve bulut sunucular için düşük gecikmeli (low-latency) çıkarım.
        """)
