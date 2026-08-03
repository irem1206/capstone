import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")
st.write("Model yükleniyor ve gerçek tahmin motoru aktif ediliyor...")

# Etiketleri doğrudan temiz liste olarak tanımlayalım (Hata riskini sıfırlar)
class_names = [
    "A Harfi", "B Harfi", "C Harfi", "Ç Harfi", "D Harfi", "E Harfi", "F Harfi", 
    "G Harfi", "Ğ Harfi", "H Harfi", "I Harfi", "İ Harfi", "J Harfi", "K Harfi", 
    "L Harfi", "M Harfi", "N Harfi", "O Harfi", "Ö Harfi", "P Harfi", "R Harfi", 
    "S Harfi", "Ş Harfi", "T Harfi", "U Harfi", "Ü Harfi", "V Harfi", "Y Harfi", "Z Harfi"
]

# Modeli güvenli bir şekilde önbelleğe alarak yükle
@st.cache_resource
def load_my_model():
    model = load_model("keras_model.h5", compile=False)
    return model

try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")
    model_loaded = False

# Oturumda bir cümle/harf geçmişi tutmak için hafıza başlatalım
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
    
    # Teachable Machine standartlarına uygun ön işleme (224x224 ve -1 ile 1 arası normalize)
    image = image.convert('RGB')
    image = np.array(image)
    img_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    img_normalized = (img_array / 127.5) - 1

    # GERÇEK TAHMİN (Model üzerinden pikseller işlenir)
    prediction = model.predict(img_normalized)
    index = np.argmax(prediction[0])
    
    # Güvenlik kontrolü: İndeks sınıf sınırları içinde mi?
    if index < len(class_names):
        class_name = class_names[index]
    else:
        class_name = "Bilinmeyen"
        
    confidence_score = float(prediction[0][index])

    st.success(f"🎯 Tahmin Edilen Harf: {class_name}")
    st.info(f"📊 Güven Oranı: %{confidence_score * 100:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ Harfi Cümleye Ekle"):
            # Sadece harf kısmını alalım (Örn: "A Harfi" -> "A")
            harf_sade = class_name.split()[0] if " " in class_name else class_name
            st.session_state.biriken_metin += harf_sade
    with col2:
        if st.button("🗑️ Cümleyi Temizle"):
            st.session_state.biriken_metin = ""

# Oluşan Cümle Paneli
st.markdown("---")
st.subheader("📝 Oluşan Cümle / Kelime Paneli")
st.text_input("Biriken Metin:", value=st.session_state.biriken_metin, disabled=True)
