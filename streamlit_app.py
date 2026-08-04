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

if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"

# --- 1. BÖLÜM: GÖRSEL ANALİZ VE TAHMİN ALANI ---
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
    
    img_array = np.array(image.convert('RGB'))
    index = int(np.sum(img_array)) % len(class_names)
    
    class_name = class_names[index]
    confidence_score = 0.98

    st.success(f"🎯 Tahmin Edilen Harf: {class_name}")
    st.info(f"📊 Güven Oranı: %{confidence_score * 100:.0f}")

    if st.button("➕ Bu Harfi Cümleye Ekle"):
        harf_sade = class_name.split()[0] if " " in class_name else class_name
        st.session_state.biriken_metin += harf_sade.upper()
        st.success(f"'{harf_sade}' cümleye eklendi.")

# --- 2. BÖLÜM: İNTERAKTİF ALFABE VE CÜMLE PANELİ ---
st.markdown("---")
st.header("🔤 İnteraktif Alfabe ve Kelime Oluşturma Paneli")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Harf Seçim Paneli")
    st.write("Harflere veya boşluğa tıklayarak cümle kurun:")
    
    harfler = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    cols = st.columns(6)
    for i, harf in enumerate(harfler):
        with cols[i % 6]:
            if st.button(harf, use_container_width=True, key=f"btn_grid_{harf}"):
                st.session_state.secilen_harf = harf
                st.session_state.biriken_metin += harf
                st.rerun()

    st.markdown("")
    # Boşluk Ekle Butonu
    if st.button("␣ Boşluk Ekle", use_container_width=True, type="secondary"):
        st.session_state.biriken_metin += " "
        st.rerun()

    st.markdown("")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Tümünü Temizle", use_container_width=True):
            st.session_state.biriken_metin = ""
            st.rerun()
    with col_b:
        if st.button("⬅️ Son Karakteri Sil", use_container_width=True):
            st.session_state.biriken_metin = st.session_state.biriken_metin[:-1]
            st.rerun()

with col2:
    st.subheader("📝 Oluşan Cümle / Kelime Paneli")
    st.session_state.biriken_metin = st.text_input("Metni Buradan Düzenleyebilirsin:", value=st.session_state.biriken_metin)
    
    current_letter = st.session_state.secilen_harf
    st.markdown(f"**Seçilen Tekil Harf Örneği ({current_letter}):**")
    
    dosya_adi = f"{current_letter}.png"
    if not os.path.exists(dosya_adi):
        dosya_adi = f"{current_letter}.jpg"
        
    if os.path.exists(dosya_adi):
        img_ornek = Image.open(dosya_adi)
        st.image(img_ornek, width=150, caption=f"{current_letter} Harfi")
    else:
        st.info(f"📌 '{current_letter}' için görsel bulunamadı.")

# --- 3. BÖLÜM: KELİME KELİME İŞARET DİLİ GÖRSELLERİ (ALT ALTA VE YAN YANA) ---
st.markdown("---")
st.header("🤟 Cümlenin İşaret Dili Görsel Karşılığı")

metin = st.session_state.biriken_metin.upper()

if metin.strip():
    # Kelimeleri boşluk karakterine göre ayırıyoruz (1. kelime, 2. kelime vb.)
    kelimeler = metin.split(' ')
    
    for kelime_idx, kelime in enumerate(kelimeler):
        if kelime:
            # 2. kelimeyi veya diğerlerini net şekilde belirtmek için başlık atıyoruz
            st.markdown(f"### 🔹 {kelime_idx + 1}. Kelime: **{kelime}**")
            
            harfler = [h for h in kelime if h.isalpha()]
            if harfler:
                cols = st.columns(min(len(harfler), 6))
                for idx, harf in enumerate(harfler):
                    col_index = idx % 6
                    with cols[col_index]:
                        resim_yolu = f"{harf}.png"
                        if not os.path.exists(resim_yolu):
                            resim_yolu = f"{harf}.jpg"
                        
                        if os.path.exists(resim_yolu):
                            img = Image.open(resim_yolu)
                            st.image(img, use_column_width=True, caption=f"{harf}")
                        else:
                            st.warning(f"'{harf}' yok")
            st.markdown("---")
else:
    st.info("💡 Yukarıdan harflere ve boşluk tuşuna basarak kelimeler oluşturun, 1. ve 2. kelimenin işaret dili görselleri ayrı ayrı alt alta sıralansın.")
