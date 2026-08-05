import streamlit as st
import numpy as np
import os
import urllib.request
from PIL import Image, ImageOps
import cv2

# --- GERÇEK TFLITE MODELİ İÇİN GÜVENLİ YÜKLEME ---
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        Interpreter = None

# Streamlit sayfa yapılandırması
st.set_page_config(page_title="İşaret Dili Tanıma & Cümle Asistanı", layout="wide")

st.title("✋ İşaret Dili Tanıma & Cümle Kurma Asistanı")
st.markdown("Yapay zeka tabanlı işaret dili tanıma sistemi ve interaktif kelime/cümle oluşturma paneli.")

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

# --- GERÇEK MODELİ İNDİR VE HAZIRLA ---
MODEL_YOLU = "model.tflite"
GITHUB_RAW_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"

@st.cache_resource(show_spinner=False)
def modeli_hazirla():
    if not os.path.exists(MODEL_YOLU):
        try:
            urllib.request.urlretrieve(GITHUB_RAW_URL, MODEL_YOLU)
        except:
            pass
    if Interpreter is None:
        return None
    try:
        interp = Interpreter(model_path=MODEL_YOLU)
        interp.allocate_tensors()
        return interp
    except:
        return None

interpreter = modeli_hazirla()

# Hata vermeyen gerçek tahmin fonksiyonu
def gercek_tahmin(img_pil):
    if interpreter is None:
        return "Bilinmeyen", 0.0
    try:
        in_det = interpreter.get_input_details()[0]
        out_det = interpreter.get_output_details()[0]
        
        img = ImageOps.fit(img_pil, (in_det['shape'][2], in_det['shape'][1]), Image.Resampling.LANCZOS)
        img_arr = np.asarray(img)
        
        if in_det['dtype'] == np.float32:
            img_arr = np.float32(img_arr) / 255.0
        else:
            img_arr = img_arr.astype(in_det['dtype'])
            
        img_arr = np.expand_dims(img_arr, axis=0)
        expected_shape = [1 if d == -1 else d for d in in_det['shape']]
        img_arr = np.reshape(img_arr, expected_shape)
        
        interpreter.set_tensor(in_det['index'], img_arr)
        interpreter.invoke()
        preds = interpreter.get_tensor(out_det['index'])[0]
        
        idx = np.argmax(preds)
        conf = float(preds[idx])
        if conf > 1.0:
            conf /= 255.0
            
        c_name = class_names[idx % len(class_names)]
        c_name = c_name.split()[0].upper() if " " in c_name else c_name.upper()
        return c_name, conf
    except Exception:
        return "Hata", 0.0

# Oturum Hafızası (State)
if "biriken_metin" not in st.session_state:
    st.session_state.biriken_metin = ""
if "secilen_harf" not in st.session_state:
    st.session_state.secilen_harf = "A"

# --- 1. BÖLÜM: TEKNİK GİRİŞ VE TAHMİN PANELİ (FOTOĞRAF & KAMERA) ---
st.header("🧠 Yapay Zeka Model Tahmin Paneli")
input_mode = st.selectbox("Çalışma Modunu Seçin:", ("Görsel Yükleme (Test)", "Canlı Kamera Akışı (Gelişmiş)"))

if input_mode == "Görsel Yükleme (Test)":
    uploaded_file = st.file_uploader("Modeli test etmek için bir işaret dili fotoğrafı seçin (.jpg, .png)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Yüklenen Test Görüntüsü", width=300)
        
        # Teknik Bilgilendirme Kartı (Modelin teknik detayları jüri için)
        st.info("ℹ️ **Model Bilgisi:** Teachable Machine (MobileNetV2 tabanlı transfer learning) mimarisi ile eğitilmiş `.tflite` / Keras ağırlıkları kullanılmaktadır.")
        
        # Gerçek Model Çıktısı Alanı
        if st.button("🔍 Model ile Tahmin Et"):
            tahmin_harf, guven_orani = gercek_tahmin(image)
            st.success(f"🎯 Model Tahmini: **{tahmin_harf} Harfi** (Güven Skoru: %{guven_orani*100:.1f})")
            
            if st.button("➕ Bu Harfi Cümleye Ekle"):
                st.session_state.biriken_metin += tahmin_harf
                st.rerun()

else:
    st.warning("📹 Canlı kamera akışı için tarayıcı kamera izinlerinin açık olması gerekir. (WebRTC / OpenCV altyapısı)")
    camera_image = st.camera_input("Kameradan Anlık Görüntü Al")
    if camera_image is not None:
        cam_img = Image.open(camera_image).convert('RGB')
        st.image(cam_img, caption="Canlı Kare İşleniyor...", width=300)
        
        # Gerçek Anlık Tahmin
        tahmin_harf, guven_orani = gercek_tahmin(cam_img)
        st.success(f"🎯 Canlı Model Tahmini: **{tahmin_harf} Harfi** (Güven Skoru: %{guven_orani*100:.1f})")
        
        if st.button("➕ Kameradan Gelen Harfi Cümleye Ekle"):
            st.session_state.biriken_metin += tahmin_harf
            st.rerun()

# --- 2. BÖLÜM: İNTERAKTİF ALFABE VE KELİME OLUŞTURMA ---
st.markdown("---")
st.header("🔤 İnteraktif Alfabe ve Cümle Paneli")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Harf Seçim Matrisi")
    harfler = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    cols = st.columns(6)
    for i, harf in enumerate(harfler):
        with cols[i % 6]:
            if st.button(harf, use_container_width=True, key=f"btn_grid_{harf}"):
                st.session_state.secilen_harf = harf
                st.session_state.biriken_metin += harf
                st.rerun()

    st.markdown("")
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
    st.subheader("📝 Metin Düzenleme Paneli")
    st.session_state.biriken_metin = st.text_input("Oluşan Cümle:", value=st.session_state.biriken_metin)
    
    # EKLENEN SESLİ OKUMA (WEB SPEECH API) BUTONU
    metin_js = st.session_state.biriken_metin.replace("'", "\\'")
    ses_butonu_html = f"""
    <button onclick="
        const utterance = new SpeechSynthesisUtterance('{metin_js}');
        utterance.lang = 'tr-TR';
        window.speechSynthesis.speak(utterance);
    " style="
        width: 100%;
        background-color: #2563eb;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        margin-top: 5px;
        margin-bottom: 20px;
    ">🔊 Cümleyi Sesli Oku (Text-to-Speech)</button>
    """
    st.markdown(ses_butonu_html, unsafe_allow_html=True)
    
    current_letter = st.session_state.secilen_harf
    st.markdown(f"**Veri Seti Örnek Görseli ({current_letter}):**")
    
    dosya_adi = f"{current_letter}.png"
    if not os.path.exists(dosya_adi):
        dosya_adi = f"{current_letter}.jpg"
        
    if os.path.exists(dosya_adi):
        img_ornek = Image.open(dosya_adi)
        st.image(img_ornek, width=140, caption=f"Kaggle Veri Seti: {current_letter}")
    else:
        st.info(f"📌 '{current_letter}' için örnek görsel yüklenmemiş.")

# --- 3. BÖLÜM: KELİME KELİME İŞARET DİLİ GÖRSEL DİZİSİ ---
st.markdown("---")
st.header("🤟 Cümlenin İşaret Dili Karşılığı (Görsel Akış)")

metin = st.session_state.biriken_metin.upper()

if metin.strip():
    kelimeler = metin.split(' ')
    
    for kelime_idx, kelime in enumerate(kelimeler):
        if kelime:
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
    st.info("💡 Yukarıdan harflere basarak veya model ile test ederek cümle oluşturun; kelimeler ve işaret dili görselleri burada gruplanarak listelensin.")
