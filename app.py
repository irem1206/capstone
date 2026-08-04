import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import urllib.request

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="İşaret Dili Akıllı Çeviri Sistemi",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ÖZEL CSS STİLLERİ (Kurumsal & Profesyonel Görünüm) ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# --- BAŞLIK VE PROJE KÜNYESİ ---
st.title("🤖 Yapay Zeka Destekli İşaret Dili Çeviri ve Cümle Sentezleme Asistanı")
st.markdown("---")

# --- MODEL VE KAYNAK YÖNETİMİ ---
MODEL_YOLU = "model.tflite"
ETIKET_YOLU = "labels.txt"
GITHUB_RAW_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"

@st.cache_resource(show_spinner=False)
def sistem_bilesenlerini_yukle():
    """Modeli ve etiketleri bellenimde önbelleğe alarak optimize eder."""
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
            return None, [], "Etiket dosyası (labels.txt) bulunamadı."
    except Exception as e:
        return None, [], fKanal Hatası: {e}

interpreter, class_names, hata_mesaji = sistem_bilesenlerini_yukle()

if hata_mesaji:
    st.error(f"Sistem Başlatılamadı: {hata_mesaji}")
else:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2])

    def tahmin_uret(frame):
        """Görüntüyü modele uygun forma getiripinference çalıştırır."""
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

    # --- ANA KONTROL PANELİ (LAYOUT) ---
    col_sol, col_sag = st.columns([1.2, 1], gap="large")

    with col_sol:
        st.subheader("📹 Canlı Veri Akışı (Girdi)")
        camera_input = st.camera_input("Kamera Sensörünü Aktifleştir")

    with col_sag:
        st.subheader("📊 Model Çıkarım & Analiz Paneli")
        sonuc_alani = st.empty()
        guven_alani = st.empty()

    if camera_input is not None:
        bytes_data = camera_input.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        class_name, confidence = tahmin_uret(frame)
        harf_sade = class_name.split()[0].upper() if " " in class_name else class_name.upper()
        
        # Sonuç Görselleştirme
        with sonuc_alani.container():
            st.markdown(f"""
                <div style='background: white; padding: 20px; border-radius: 10px; border-left: 6px solid #28a745; text-align: center;'>
                    <h3 style='margin:0; color: #333;'>Tespit Edilen Sınıf</h3>
                    <h1 style='margin:10px 0; color: #28a745; font-size: 3em;'>{harf_sade}</h1>
                    <p style='margin:0; color: #666; font-size: 0.9em;'>Ham Etiket: {class_name}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with guven_alani.container():
            st.markdown("**Model Güven Skoru (Confidence Score):**")
            st.progress(confidence)
            st.caption(f"Doğruluk Oranı: %{confidence * 100:.2f}")

        # --- METİN VE CÜMLE SENTEZLEME MOTORU ---
        st.markdown("---")
        st.subheader("📝 Metin ve Cümle Sentezleme")
        
        if 'cumle_hafizasi' not in st.session_state:
            st.session_state.cumle_hafizasi = ""
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("➕ Karakteri Cümleye Ekle", type="primary"):
                st.session_state.cumle_hafizasi += harf_sade
        with col_btn2:
            if st.button("␣ Boşluk Ekle"):
                st.session_state.cumle_hafizasi += " "
                st.rerun()

        st.session_state.cumle_hafizasi = st.text_input("Oluşan Anlamlı Metin Çıktısı:", value=st.session_state.cumle_hafizasi)

        col_islem1, col_islem2 = st.columns(2)
        with col_islem1:
            if st.button("⬅️ Son Karakteri Sil"):
                st.session_state.cumle_hafizasi = st.session_state.cumle_hafizasi[:-1]
                st.rerun()
        with col_islem2:
            if st.button("🧹 Belleği Temizle"):
                st.session_state.cumle_hafizasi = ""
                st.rerun()
    else:
        with col_sag:
            st.info("💡 Analiz başlatmak için lütfen sol taraftaki kameradan bir kare yakalayın.")

    # --- JÜRİ İÇİN TEKNİK BİLGİ SEKMESİ (Sistem Mimarisi) ---
    with st.expander("⚙️ Jüri ve Teknik Detaylar Bilgi Kartı"):
        st.markdown(f"""
        - **Kullanılan Mimari:** TensorFlow Lite (TFLite) Optimize Edilmiş Edge-AI Modeli
        - **Giriş Çözünürlüğü:** {target_size[0]}x{target_size[1]} piksel RGB Tensor Matrisi
        - **Bellek Yönetimi:** `st.cache_resource` ile donanım katmanı önbelleklemesi aktif.
        - **Çıkarım Süresi (Inference Latency):** Düşük gecikmeli CPU/XNNPACK donanım ivmelenmesi.
        """)
