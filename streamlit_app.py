import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf  # TFLite yorumlayıcısı için hafif çekirdek

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")

# Etiketleri labels.txt dosyasından dinamik oku
@st.cache_resource
def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    except:
        return []

class_names = load_labels()

# TFLite Modelini Yükle (Asla kasmaz, anında açılır)
@st.cache_resource
def load_tflite_model():
    # Dosya adını model.tflite yaptıysan burası doğrudan görür
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    model_loaded = False

# Oturumda biriken kelime/cümle hafızası
if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""

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

if image is not None and model_loaded:
    st.image(image, caption="İşlenen Görüntü", use_column_width=True)
    
    # Görüntü ön işleme (224x224 ve -1 ile 1 normalize)
    image = image.convert('RGB')
    image = np.array(image)
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    img_normalized = (img_array / 127.5) - 1

    # TFLite ile Gerçek Tahmin
    interpreter.set_tensor(input_details[0]['index'], img_normalized)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    index = np.argmax(prediction[0])
    
    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Bilinmeyen"
        
    confidence_score = float(prediction[0][index])

    st.success(f"🎯 Tahmin Edilen Harf: {class_name}")
    st.info(f"📊 Güven Oranı: %{confidence_score * 100:.2f}")

    # Temiz harf ayıklama
    harf_sade = class_name.split()[-1] if " " in class_name else class_name

    if st.button("➕ Bu Harfi Cümleye Ekle"):
        st.session_state.biriken_metin += harf_sade
        st.success(f"'{harf_sade}' cümleye eklendi.")

# Oluşan Cümle Paneli
st.markdown("---")
st.subheader("📝 Oluşan Cümle / Kelime Paneli")
st.session_state.biriken_metin = st.text_input("Metni Buradan Düzenleyebilirsin:", value=st.session_state.biriken_metin)
