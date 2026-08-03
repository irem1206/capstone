import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.title("✋ İşaret Dili Tanıma Asistanı")
st.write("Kameranızı kullanabilir veya fotoğraf yükleyebilirsiniz.")

@st.cache_resource
def get_model():
    model = load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = get_model()

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

if image is not None:
    # Görüntüyü ekranda göster
    st.image(image, caption="İşlenen Görüntü", use_column_width=True)
    
    # Modeli tahmin için hazırla
    image = image.convert('RGB')
    image = np.array(image)
    img_resized = cv2.resize(image, (224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    img_normalized = (img_array / 127.5) - 1

    prediction = model.predict(img_normalized)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = float(prediction[0][index])

    st.success(f"🎯 Tahmin: {class_name}")
    st.info(f"📊 Güven Oranı: %{confidence_score * 100:.2f}")
