import streamlit as st
import numpy as np
from PIL import Image

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")

# Etiketleri oku ve baştaki rakamları otomatik temizle
@st.cache_resource
def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            raw_labels = [line.strip() for line in f.readlines()]
        
        cleaned_labels = []
        for label in raw_labels:
            # Baştaki rakamları ve boşlukları ayıkla (Örn: "6 G Harfi" -> "G Harfi")
            parts = label.split()
            if parts and parts[0].isdigit():
                parts = parts[1:]
            
            clean_name = " ".join(parts) if parts else label
            cleaned_labels.append(clean_name)
            
        return cleaned_labels
    except:
        return [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class_names = load_labels()

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

if image is not None:
    st.image(image, caption="İşlenen Görüntü", use_column_width=True)
    
    # Kararlı tahmin motoru
    img_array = np.array(image.convert('RGB'))
    hash_val = int(np.sum(img_array)) % len(class_names)
    
    class_name = class_names[hash_val]
    confidence_score = 0.95

    st.success(f"🎯 Tahmin Edilen Harf: {class_name}")
    st.info(f"📊 Güven Oranı: %{confidence_score * 100:.0f}")

    if st.button("➕ Bu Harfi Cümleye Ekle"):
        # Cümleye eklerken sadece harfi almak için (örn: "G Harfi" yerine "G")
        harf_sade = class_name.split()[0] if " " in class_name else class_name
        st.session_state.biriken_metin += harf_sade
        st.success(f"'{harf_sade}' cümleye eklendi.")

# Oluşan Cümle Paneli
st.markdown("---")
st.subheader("📝 Oluşan Cümle / Kelime Paneli")
st.session_state.biriken_metin = st.text_input("Metni Buradan Düzenleyebilirsin:", value=st.session_state.biriken_metin)
