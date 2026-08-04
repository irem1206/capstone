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

# --- PROFESYONEL ÜST KISIM (HERO SECTION) ---
st.markdown("""
    <div class="hero-container">
        <p class="hero-title">🤟 Yapay Zeka Destekli Türk İşaret Dili Çeviri Asistanı</p>
        <p class="hero-subtitle">Edge-AI Tabanlı Gerçek Zamanlı Görsel Tanıma ve Dinamik Cümle Sentezleme Motoru</p>
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

    # --- GİRDİ YÖNTEMİ SEÇİMİ (Fotoğraf Yükle / Kamera Kullan) ---
    girdi_turu = st.radio("Girdi türünü seçin:", ["Fotoğraf Yükle", "Kamera Kullan"], horizontal=True)

    frame = None

    if girdi_turu == "Fotoğraf Yükle":
        yuklenen_dosya = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])
        if yuklenen_dosya is not None:
            image = Image.open(yuklenen_dosya).convert('RGB')
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        kamera_girdisi = st.camera_input("Kameradan Anlık Görüntü Yakala")
        if kamera_girdisi is not None:
            bytes_data = kamera_girdisi.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # --- ANA YERLEŞİM (LAYOUT) ---
    col_kamera, col_analiz = st.columns([1.2, 1], gap="large")

    with col_analiz:
        st.subheader("📊 Model Çıkarım & Doğruluk Filtresi")
        sonuc_alani = st.empty()
        guven_alani = st.empty()

    if frame is not None:
        class_name, confidence = tahmin_uret(frame)
        harf_sade = class_name.split()[0].upper() if " " in class_name else class_name.upper()
        
        # --- KRİTİK DOĞRULUK FİLTRESİ (%70 Eşiği) ---
        GUVEN_ESIGI = 0.70  
        
        if confidence < GUVEN_ESIGI:
            with sonuc_alani.container():
                st.markdown(f"""
                    <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #ef4444; text-align: center;'>
                        <h3 style='margin:0; color: #ef4444;'>⚠️ Düşük Güven Skoru</h3>
                        <h2 style='margin:10px 0; color: #f87171;'>Model Kararsız Kaldı</h2>
                        <p style='margin:0; color: #9ca3af;'>Tespit Edilen: {harf_sade} (Güven: %{confidence * 100:.1f})</p>
                        <p style='margin-top:5px; color: #fbbf24; font-size:0.85em;'>Lütfen görüntüyü daha net konumlandırıp tekrar deneyin.</p>
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
            st.caption(f"Doğruluk Oranı: %{confidence * 100:.2f} (Eşik Sınırı: %{int(GUVEN_ESIGI*100)})")

        # --- CÜMLE VE METİN SENTEZLEME BİRİMİ ---
        st.markdown("---")
        st.subheader("📝 Dinamik Cümle Sentezleme Motoru")
        
        if 'cumle_hafizasi' not in st.session_state:
            st.session_state.cumle_hafizasi = ""
            
        col_islem1, col_islem2 = st.columns(2)
        with col_islem1:
            if st.button("➕ Harfi Cümleye Aktar", type="primary"):
                if harf_eklenebilir:
                    st.session_state.cumle_hafizasi += harf_sade
                else:
                    st.warning("Model güven eşiğinin altında olduğu için bu harf eklenmedi!")
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
            if st.button("🧹 Belleği Sıfırla"):
                st.session_state.cumle_hafizasi = ""
                st.rerun()

        # --- ALT KISIM: SANAL KLAVYE / HARF SEÇİM PANELİ (X, Q, W Hariç) ---
        st.markdown("---")
        st.subheader("⌨️ Manuel Harf Giriş Paneli (Sanal Klavye)")
        st.markdown("<p style='color: #9ca3af; font-size: 0.9em;'>Harflere tıklayarak da kelime oluşturabilirsiniz:</p>", unsafe_allow_html=True)

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
                    if st.button(harf, key=f"klavye_{harf}"):
                        st.session_state.cumle_hafizasi += harf
                        st.rerun()

    else:
        with col_analiz:
            st.info("👈 Analizi başlatmak ve tahmin sonuçlarını görmek için yukarıdan bir girdi yöntemi seçin.")

   
