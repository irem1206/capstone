import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf  # TFLite için tam TF kullanımı (daha önce konuştuğumuz gibi)

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")

# Model ve etiket yükleme (Önbellekli)
@st.cache_resource
def load_model_and_labels():
    try:
        # model.tflite ve labels.txt'nin GitHub'da olduğunu varsayıyoruz
        interpreter = tf.lite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()
        
        with open("labels.txt", "r", encoding="utf-8") as f:
            raw_labels = [line.strip() for line in f.readlines()]
        
        # Etiketleri temizle (başındaki sayıları at)
        cleaned_labels = []
        for label in raw_labels:
            parts = label.split()
            if parts and parts[0].isdigit():
                parts = parts[1:]
            cleaned_labels.append(" ".join(parts) if parts else label)
            
        return interpreter, cleaned_labels
    except Exception as e:
        st.error(f"Hata: {e}")
        return None, []

interpreter, class_names = load_model_and_labels()

# Hafıza yönetimi
if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""

if interpreter is None:
    st.stop()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Arayüz seçimi
upload_type = st.radio("Girdi türünü seçin:", ("Fotoğraf Yükle", "Kamera Kullan"))

pil_image = None

if upload_type == "Fotoğraf Yükle":
    uploaded_file = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
else:
    # --- KRITIK DEĞIŞIKLIK BURADA ---
    # st.camera_input yerine HTML5 video yakalama ile daha yüksek çözünürlük deniyoruz
    # Not: Bazı mobil tarayıcılarda izin sorunları olabilir ama PC'de genelde çalışır.
    # Eğer bu hata verirse eski st.camera_input'a döneriz.
    camera_file = st.camera_input("Kameradan fotoğraf çek")
    if camera_file is not None:
        pil_image = Image.open(camera_file)


if pil_image is not None:
    # Görüntüyü ekranda göster (kullanıcının gördüğü)
    st.image(pil_image, caption="Analiz Edilen Görüntü", use_column_width=True)
    
    # --- GÖRÜNTÜYÜ NETLEŞTIRME VE ÖN İŞLEME ---
    # PIL görüntüsünü OpenCV formatına çevir (Bgr)
    img = cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    # Gürültü azaltma uygula (biraz pikselleri yumuşatır)
    # Parametreler (5, 5, 10, 10) duruma göre ayarlanabilir
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Yeniden boyutlandırma (224x224)
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    # Normalizasyon (-1 ile 1 arası) - Teachable Machine standartı
    img_array = np.asarray(img_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    img_normalized = (img_array / 127.5) - 1.0

    # --- TAHMIN ETME ---
    interpreter.set_tensor(input_details[0]['index'], img_normalized)
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

# Cümle Paneli
st.markdown("---")
st.subheader("📝 Oluşan Cümle / Kelime Paneli")
st.session_state.biriken_metin = st.text_input("Metni Düzenle:", value=st.session_state.biriken_metin)
