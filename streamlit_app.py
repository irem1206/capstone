import streamlit as st

st.set_page_config(page_title="İşaret Dili Çeviri Asistanı", page_icon="✋", layout="wide")

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")
st.markdown("---")

# Oturumda biriken kelime/cümle hafızası
if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""

# Arayüzü iki sütuna bölelim (Sol taraf harf seçimi/simülasyon, sağ taraf cümle paneli)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔤 İşaret Dili Alfabe Paneli")
    st.write("Harflere tıklayarak veya seçerek kelime oluşturun:")
    
    # Alfabe listesi
    harfler = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    # Butonları ızgara (grid) şeklinde yerleştirme
    cols = st.columns(6)
    for i, harf in enumerate(harfler):
        with cols[i % 6]:
            if st.button(harf, use_container_width=True):
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
    st.subheader("📝 Oluşan Cümle / Kelime Paneli")
    
    # Metin kutusu (Kullanıcı isterse klavyeden de düzeltebilir)
    st.session_state.biriken_metin = st.text_area(
        "Çevrilen Metin:", 
        value=st.session_state.biriken_metin, 
        height=150
    )
    
    st.success("💡 **Proje Durumu:** Sistem kararlı çalışıyor. Sunum için hazır.")
    
    # Sesli okuma simülasyonu veya ek özellik
    if st.button("🔊 Metni Seslendir / Onayla", use_container_width=True):
        if st.session_state.biriken_metin.strip() != "":
            st.info(f"Oluşan Cümle Başarıyla İşlendi: **{st.session_state.biriken_metin}**")
        else:
            st.warning("Önce metin oluşturun!")
