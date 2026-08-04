import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2  # Kamera akışı için gerekli
import os
import urllib.request

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="İşaret Dili Canlı Çeviri", layout="wide")

# --- BAŞLIK VE AÇIKLAMA ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>✋ İşaret Dili Canlı Çeviri Paneli</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Modelinizi kameranın önüne koyun, anlık olarak tahmin etsin.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- MODEL VE ETİKET YÜKLEME ---
MODEL_YOLU = "model.tflite"
ETIKET_YOLU = "labels.txt"
GITHU B_RAW_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"

@st.cache_resource # Modeli önbelleğe al
def modeli_yukle():
    # Eğer model yoksa indir
    if not os.path.exists(MODEL_YOLU):
        with st.spinner("Model dosyası indiriliyor..."):
            try:
                urllib.request.urlretrieve(GITHUB_RAW_URL, MODEL_YOLU)
                st.success("Model başarıyla indirildi.")
            except Exception as e:
                st.error(f"Model indirilemedi: {e}")
                return None, []
    
    # TFLite Modelini Yükleme
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_YOLU)
        interpreter.allocate_tensors()
        
        # Etiketleri Yükleme
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
            return interpreter, class_names
        else:
            st.error(f"{ETIKET_YOLU} dosyası bulunamadı.")
            return interpreter, []

    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None, []

interpreter, class_names = modeli_yukle()

if interpreter and class_names:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    target_size = (input_shape[1], input_shape[2]) # Genelde 224x224 veya benzeri

    # --- TAHMİN FONKSİYONU ---
    def goruntuyu_isle_ve_tahmin_et(frame):
        # OpenCV karesini (BGR) PIL formatına (RGB) çevir
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Modeli giriş boyutuna göre yeniden boyutlandır ve hazırla
        img = img_pil.resize(target_size)
        img_array = np.asarray(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Batch boyutu ekle
        
        # Modeli çalıştır
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        # Sonuçları al
        index = np.argmax(prediction[0])
        confidence = float(prediction[0][index])
        class_name = class_names[index] if index < len(class_names) else "Bilinmeyen"
        
        return class_name, confidence

    # --- ARAYÜZ YERLEŞİMİ (LAYOUT) ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Canlı Kamera Akışı")
        # Streamlit'in kamera giriş bileşeni
        camera_input = st.camera_input("Kamerayı Başlat")

    with col2:
        st.subheader("🤖 Anlık Tahmin Sonucu")
        sonuc_kutusu = st.empty() # Sonuçları göstereceğimiz boş alan
        confidence_bar = st.progress(0) # Güven çubuğu

    # --- SÜREKLİ TAHMİN DÖNGÜSÜ ---
    if camera_input is not None:
        # Kamera görüntüsünü oku ve işle
        bytes_data = camera_input.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Modeli çalıştır
        class_name, confidence = goruntuyu_isle_ve_tahmin_et(frame)
        
        # Sonuçları ekrana yaz
        harf_sade = class_name.split()[0].upper() if " " in class_name else class_name.upper()
        
        with sonuc_kutusu:
             st.markdown(f"<div style='font-size: 4em; font-weight: bold; color: #2196F3; text-align: center; padding: 30px; background-color: #E3F2FD; border-radius: 10px;'>{harf_sade}</div>", unsafe_allow_html=True)
             st.caption(f"Tam İsim: {class_name}")
        
        confidence_bar.progress(confidence)
        st.info(f"Modelin bu harften emin olma oranı: %{confidence * 100:.1f}")

        # --- EKLEME VE CÜMLE YÖNETİMİ ---
        st.markdown("---")
        st.subheader("📝 Cümle Oluşturma Paneli")
        
        # Session State içinde cümleyi tut
        if 'tam_cumle' not in st.session_state:
            st.session_state.tam_cumle = ""
            
        # Kullanıcının tahmini onaylaması için buton
        if st.button(f"➕ '{harf_sade}' Harfini Cümleye Ekle", type="primary"):
            st.session_state.tam_cumle += harf_sade
        
        # Cümle gösterimi ve düzenleme alanı
        st.session_state.tam_cumle = st.text_area("Oluşan Metin", value=st.session_state.tam_cumle, height=100)
        
        # Kontrol butonları
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("␣ Boşluk Ekle"):
                st.session_state.tam_cumle += " "
                st.rerun()
        with col_b:
            if st.button("⬅️ Son Karakteri Sil"):
                st.session_state.tam_cumle = st.session_state.tam_cumle[:-1]
                st.rerun()
        if st.button("🧹 Tümünü Temizle"):
            st.session_state.tam_cumle = ""
            st.rerun()

else:
    st.warning("Uygulama başlatılamadı. Model dosyası veya etiketler yüklenemedi.")
