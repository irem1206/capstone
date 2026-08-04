import streamlit as st
import numpy as np
import os
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="İşaret Dili Tanıma & Cümle Kurma Asistanı", layout="wide")

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")

# Model ve etiketleri yükle (Önbellekli)
@st.cache_resource
def load_model_and_labels():
    try:
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        
        with open("labels.txt", "r", encoding="utf-8") as f:
            raw_labels = [line.strip() for line in f.readlines()]
        
        cleaned_labels = []
        for label in raw_labels:
            parts = label.split()
            if parts and parts[0].isdigit():
                parts = parts[1:]
            clean_name = " ".join(parts) if parts else label
            cleaned_labels.append(clean_name)
            
        return interpreter, cleaned_labels
    except Exception as e:
        return None, []

interpreter, class_names = load_labels()

if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"

# --- 1. BÖLÜM: GERÇEK TFLITE MODELİ İLE TAHMİN ALANI ---
upload_type = st.radio("Girdi türünü seçin:", ("Fotoğraf Yükle", "Kamera Kullan"))

image = None

if upload_type == "Fotoğraf Yükle":
    uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Kameradan fotoğraf çek")
    if camera_file is not None:
        image = Image.open(camera_file)

if image is not None and interpreter is not None:
    st.image(image, caption="İşlenen Görüntü", use_column_width=True)
    
    # Modelin beklediği standart boyut (224x224) ve normalizasyon
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img_resized = image.resize((224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    index = np.argmax(prediction[0])
    class_name = class_names[index] if index < len(class_names) else "Bilinmeyen"
    confidence_score = float(prediction[0][index])

    st.success(f"🎯 Tahmin Edilen Harf: {class_name}")
    st.info(f"📊 Güven Oranı: %{confidence_score * 100:.2f}")

    if st.button("➕ Bu Harfi Cümleye Ekle"):
        harf_sade = class_name.split()[0] if " " in class_name else class_name
        st.session_state.biriken_metin += harf_sade
        st.success(f"'{harf_sade}' cümleye eklendi.")

# --- 2. BÖLÜM: İNTERAKTİF ALFABE VE GÖRSEL PANELİ ---
st.markdown("---")
st.header("🔤 İnteraktif Alfabe ve Veri Seti Görsel Paneli")
st.write("Aşağıdaki harflere tıklayarak hem Kaggle veri seti örneğini sağda görebilir hem de cümleye ekleyebilirsiniz:")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Harf Seçim Paneli")
    harfler = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    cols = st.columns(6)
    for i, harf in enumerate(harfler):
        with cols[i % 6]:
            if st.button(harf, use_container_width=True, key=f"btn_grid_{harf}"):
                st.session_state.secilen_harf = harf
                st.session_state.biriken_metin += harf
                st.rerun()

    st.markdown("")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Tümünü Temizle", use_container_width=True):
            st.session_state.biriken_metin = ""
            st.rerun()
    with col_b:
        if st.button("⬅️ Son Harfi Sil", use_container_width=True):
            st.session_state.biriken_metin = st.session_state.biriken_metin[:-1]
            st.rerun()

with col2:
    current_letter = st.session_state.secilen_harf
    st.subheader(f"🖼️ '{current_letter}' Veri Seti Örneği")
    
    dosya_adi = f"{current_letter}.png"
    if not os.path.exists(dosya_adi):
        dosya_adi = f"{current_letter}.jpg"
        
    if os.path.exists(dosya_adi):
        try:
            img_ornek = Image.open(dosya_adi)
            st.image(img_ornek, caption=f"Kaggle Veri Seti Örneği: {current_letter} Harfi", use_column_width=True)
        except:
            st.info(f"📌 Seçilen Harf: **{current_letter}**")
    else:
        st.info(f"📌 Seçilen Harf: **{current_letter}**")

# --- 3. BÖLÜM: OLUŞAN CÜMLE PANELİ ---
st.markdown("---")
st.subheader("📝 Oluşan Cümle / Kelime Paneli")
st.session_state.biriken_metin = st.text_input("Metni Buradan Düzenleyebilirsin:", value=st.session_state.biriken_metin)
