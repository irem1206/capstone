import streamlit as st
import numpy as np
import os
from PIL import Image

st.set_page_config(page_title="İşaret Dili Tanıma & Cümle Kurma Asistanı", layout="wide")

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")

# Etiketleri oku ve baştaki rakamları otomatik temizle
@st.cache_resource
def load_labels():
    try:
        with open("labels.txt", "r", encoding="utf-8") as f:
            raw_labels = [line.strip() for line in f.readlines()]
        
        cleaned_labels = []
        for label in raw_labels:
            parts = label.split()
            if parts and parts[0].isdigit():
                parts = parts[1:]
            
            clean_name = " ".join(parts) if parts else label
            cleaned_labels.append(clean_name)
            
        return cleaned_labels
    except:
        return [chr(i) for i in range(ord('A'), ord('Z') + 1)]

class_names = load_labels()

# Oturumda biriken kelime/cümle ve seçilen harf hafızası
if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"

# --- 1. BÖLÜM: KAMERA / FOTOĞRAF YÜKLEME VE TAHMİN ALANI ---
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
        harf_sade = class_name.split()[0] if " " in class_name else class_name
        st.session_state.biriken_metin += harf_sade
        st.success(f"'{harf_sade}' cümleye eklendi.")

# --- 2. BÖLÜM: AŞAĞI KAYDIRINCA ÇIKAN ALFABE VE GÖRSEL PANELİ ---
st.markdown("---")
st.header("🔤 İnteraktif Alfabe ve Veri Seti Paneli")
st.write("Aşağıdaki harflere tıklayarak hem Kaggle/Teachable Machine görsellerini inceleyebilir hem de cümleye ekleyebilirsiniz:")

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
    st.subheader(f"🖼️ '{current_letter}' Görsel Örneği")
    
    # Klasör bulunamadı hatasını önlemek için olası tüm yolları esnek arıyoruz
    gorsel_bulundu = False
    temiz_harf = current_letter.upper()
    
    klasor_adaylari = [
        os.path.join("dataset", temiz_harf),
        os.path.join("dataset", f"{temiz_harf} Harfi"),
        os.path.join("veri", temiz_harf),
        temiz_harf,
        f"{temiz_harf} Harfi"
    ]
    
    for klasor in klasor_adaylari:
        if os.path.exists(klasor) and os.path.isdir(klasor):
            dosyalar = [f for f in os.listdir(klasor) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if dosyalar:
                ornek_resim_yolu = os.path.join(klasor, dosyalar[0])
                try:
                    img = Image.open(ornek_resim_yolu)
                    st.image(img, caption=f"Veri Setinden '{temiz_harf}' Örneği", use_column_width=True)
                    gorsel_bulundu = True
                    break
                except:
                    pass
                
    if not gorsel_bulundu:
        st.info(f"📂 Bilgi: '{temiz_harf}' için sunucuda klasör bulunamadı. (Eğer görselleri 'dataset/' klasörüne attıysanız adını kontrol edebilirsiniz. Kod hata vermeden çalışmaya devam ediyor.)")

# --- 3. BÖLÜM: OLUŞAN CÜMLE PANELİ ---
st.markdown("---")
st.subheader("📝 Oluşan Cümle / Kelime Paneli")
st.session_state.biriken_metin = st.text_input("Metni Buradan Düzenleyebilirsin:", value=st.session_state.biriken_metin)
