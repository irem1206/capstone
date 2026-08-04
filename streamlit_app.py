import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="İşaret Dili Tanıma & Cümle Kurma Asistanı", page_icon="✋", layout="wide")

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")
st.markdown("---")

# Oturumda biriken kelime/cümle hafızası ve seçilen harf
if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"

# Arayüzü iki sütuna bölelim (Sol taraf harf seçimi, sağ taraf Teachable Machine görsel paneli)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔤 İşaret Dili Alfabe Paneli")
    st.write("Harflere tıklayarak hem cümleye ekle hem sağda eğitim görselini gör:")
    
    # Alfabe listesi (Z harfine kadar)
    harfler = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    # Butonları ızgara (grid) şeklinde yerleştirme
    cols = st.columns(6)
    for i, harf in enumerate(harfler):
        with cols[i % 6]:
            if st.button(harf, use_container_width=True, key=f"btn_{harf}"):
                st.session_state.secilen_harf = harf
                st.session_state.biriken_metin += harf
                st.rerun()

    st.markdown("---")
    if st.button("🧹 Tüm Metni Temizle", type="primary", use_container_width=True):
        st.session_state.biriken_metin = ""
        st.rerun()
        
    if st.button("⬅️ Son Harfi Sil", use_container_width=True):
        st.session_state.biriken_metin = st.session_state.biriken_metin[:-1]
        st.rerun()

with col2:
    current_letter = st.session_state.secilen_harf
    st.subheader(f"🖼️ Teachable Machine / Veri Seti Görseli: '{current_letter}'")
    
    # Proje klasöründe her harf için bir klasör olduğunu varsayıyoruz (Örn: dataset/A/ veya dataset/A Harfi/)
    # Eğer görseller klasörlerindeyse otomatik çeker, yoksa bilgilendirir.
    gorsel_bulundu = False
    
    # Olası klasör isimleri varyasyonları
    klasor_adaylari = [
        os.path.join("dataset", current_letter),
        os.path.join("dataset", f"{current_letter} Harfi"),
        current_letter,
        f"{current_letter} Harfi"
    ]
    
    for klasor in klasor_adaylari:
        if os.path.exists(klasor) and os.path.isdir(klasor):
            dosyalar = [f for f in os.listdir(klasor) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            if dosyalar:
                # İlk görseli örnek olarak gösterelim
                ornek_resim_yolu = os.path.join(klasor, dosyalar[0])
                img = Image.open(ornek_resim_yolu)
                st.image(img, caption=f"Kaggle Veri Setinden '{current_letter}' Örneği", use_column_width=True)
                gorsel_bulundu = True
                break
                
    if not gorsel_bulundu:
        st.info(f"'{current_letter}' harfi için veri seti klasöründe görsel bulunamadı. (Klasör adını kontrol edebilirsin)")

    st.markdown("---")
    st.subheader("📝 Oluşan Cümle / Kelime Paneli")
    st.session_state.biriken_metin = st.text_input(
        "Çevrilen Metin:", 
        value=st.session_state.biriken_metin
    )
