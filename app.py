import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import urllib.request
from PIL import Image, ImageOps

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="İşaret Dili Akıllı Çeviri ve Sentezleme",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN KURUMSAL STİLLER ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .hero-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #374151;
        text-align: center;
        margin-bottom: 25px;
    }
    .hero-title { color: #60a5fa; font-size: 2.2em; font-weight: 800; margin: 0; }
    .hero-subtitle { color: #9ca3af; font-size: 1.1em; margin-top: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2563eb; color: white; border: none; }
    .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

# --- ÜST KISIM (HERO SECTION) ---
st.markdown("""
    <div class="hero-container">
        <p class="hero-title">🤟 Yapay Zeka Destekli Türk İşaret Dili Çeviri Asistanı</p>
        <p class="hero-subtitle">Edge-AI Tabanlı Gerçek Zamanlı Görsel Tanıma ve Dinamik Cümle Sentezleme Motoru</p>
    </div>
""", unsafe_allow_html=True)

# --- HAFIZA VE DURUM YÖNETİMİ ---
if 'cumle_hafizasi' not in st.session_state:
    st.session_state.cumle_hafizasi = ""
if 'gorsel_harf' not in st.session_state:
    st.session_state.gorsel_harf = ""  # Kameradan veya klavyeden gelen anlık harf
if 'gorsel_guven' not in st.session_state:
    st.session_state.gorsel_guven = 0.0

# --- MODEL VE KAYNAK YÖNETİMİ ---
MODEL_YOLU = "model.tflite"
ETIKET_YOLU = "labels.txt"
GITHUB_RAW_URL = "https://github.com/irem1206/capstone/raw/refs/heads/main/model.tflite"

@st.cache_resource(show_spinner=False)
def model_ve_etiketleri_yukle():
    if not os.path.exists(MODEL_YOLU):
        try:
            urllib.request.urlretrieve(GITHUB_RAW_URL, MODEL_YOLU)
        except:
            pass
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_YOLU)
        interpreter.allocate_tensors()
        class_names = []
        if os.path.exists(ETIKET_YOLU):
            with open(ETIKET_YOLU, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        parts = parts[1:]
                    class_names.append(" ".join(parts) if parts else line.strip())
        return interpreter, class_names
    except:
        return None, []

interpreter, class_names = model_ve_etiketleri_yukle()

def tahmin_uret(frame):
    if not interpreter:
        return "Bilinmeyen", 0.0
    
    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    target_h, target_w = in_det['shape'][1], in_det['shape'][2]
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (target_w, target_h))
    
    # Boyut ve tip hatalarını (ValueError) kökünden çözen dönüşüm
    img_arr = np.asarray(img_resized, dtype=np.float32) / 255.0
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
    c_name = class_names[idx] if idx < len(class_names) else "Bilinmeyen"
    return c_name, conf

# --- ANA EKRAN DÜZENİ ---
col_sol, col_sag = st.columns([1.2, 1], gap="large")

with col_sol:
    st.subheader("📹 Canlı Veri Akışı (Girdi)")
    girdi_turu = st.radio("Girdi türünü seçin:", ["Fotoğraf Yükle", "Kamera Kullan"], horizontal=True)
    frame = None
    
    if girdi_turu == "Fotoğraf Yükle":
        dosya = st.file_uploader("Bir resim seçin...", type=["jpg", "jpeg", "png"])
        if dosya:
            file_bytes = np.asarray(bytearray(dosya.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        kamera = st.camera_input("Kamera Sensörünü Aktifleştir")
        if kamera:
            bytes_data = kamera.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

# Modeli sadece geçerli bir görüntü (frame) varsa çalıştır
if frame is not None:
    try:
        h, c = tahmin_uret(frame)
        st.session_state.gorsel_harf = h.split()[0].upper()
        st.session_state.gorsel_guven = c
    except Exception:
        pass

with col_sag:
    st.subheader("📊 Model Çıkarım & Doğruluk Filtresi")
    
    # Ne kamera ne de klavye kullanılmadıysa başlangıç uyarısını göster
    if st.session_state.gorsel_harf == "":
        st.info("👈 Analizi başlatmak veya görsel paneli görmek için soldan bir resim ekleyin ya da en aşağıdaki klavyeden bir harfe tıklayın.")
    else:
        harf = st.session_state.gorsel_harf
        guven = st.session_state.gorsel_guven
        
        # Sadece model tahminlerinde güven %70 altındaysa uyar (Klavyede güven hep 1.0'dır)
        if guven < 0.70 and frame is not None:
            st.markdown(f"""
                <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #ef4444; text-align: center;'>
                    <h3 style='margin:0; color: #ef4444;'>⚠️ Düşük Güven Skoru</h3>
                    <h2 style='margin:10px 0; color: #f87171;'>Model Kararsız Kaldı</h2>
                    <p style='margin:0; color: #9ca3af;'>Tespit Edilen: {harf} (Güven: %{guven * 100:.1f})</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #10b981; text-align: center;'>
                    <span style='color: #9ca3af; font-size: 0.9em; text-transform: uppercase;'>Aktif Tahmin / Seçim</span>
                    <h1 style='margin: 10px 0; color: #34d399; font-size: 4em; font-weight: 800;'>{harf}</h1>
                    <span style='color: #9ca3af; font-size: 0.85em;'>Güven Skoru: %{guven * 100:.1f}</span>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(guven)

# --- CÜMLE VE SES SENTEZLEME BİRİMİ ---
st.markdown("---")
st.subheader("📝 Dinamik Cümle ve Ses Sentezleme Motoru")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    # Eğer yeşil ekranda bir harf varsa (kamera veya klavye yoluyla) cümleye ekler
    if st.button("➕ Görseldeki Harfi Cümleye Aktar", type="primary"):
        if st.session_state.gorsel_harf:
            st.session_state.cumle_hafizasi += st.session_state.gorsel_harf
            st.rerun()
with col_btn2:
    if st.button("␣ Boşluk Karakteri Ekle"):
        st.session_state.cumle_hafizasi += " "
        st.rerun()

st.session_state.cumle_hafizasi = st.text_input("Oluşan Anlamlı Metin Çıktısı:", value=st.session_state.cumle_hafizasi)

# --- TARAYICI TABANLI SESLİ OKUMA ---
metin_js = st.session_state.cumle_hafizasi.replace("'", "\\'")
st.markdown(f"""
<button onclick="
    const utterance = new SpeechSynthesisUtterance('{metin_js}');
    utterance.lang = 'tr-TR';
    window.speechSynthesis.speak(utterance);
" style="width: 100%; background-color: #2563eb; color: white; padding: 10px 20px; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 10px;">
🔊 Cümleyi Sesli Oku (Text-to-Speech)
</button>
""", unsafe_allow_html=True)

col_del1, col_del2 = st.columns(2)
with col_del1:
    if st.button("⬅️ Son Karakteri Sil"):
        st.session_state.cumle_hafizasi = st.session_state.cumle_hafizasi[:-1]
        st.rerun()
with col_del2:
    if st.button("🧹 Belleği Sıfırla"):
        st.session_state.cumle_hafizasi = ""
        st.rerun()

# --- SANAL KLAVYE ---
st.markdown("---")
st.subheader("⌨️ Manuel Harf Giriş Paneli (Sanal Klavye)")
st.markdown("<p style='color: #9ca3af; font-size: 0.9em;'>Harflere tıkladığınızda yukarıdaki yeşil panel anında o harfe güncellenir ve kelimeye eklenir:</p>", unsafe_allow_html=True)

alfabe = [
    ["A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ"],
    ["H", "I", "İ", "J", "K", "L", "M", "N", "O"],
    ["Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V"],
    ["Y", "Z"]
]

for satir in alfabe:
    c = st.columns(len(satir))
    for i, h in enumerate(satir):
        with c[i]:
            if st.button(h, key=f"klavye_{h}"):
                # Klavyeye basıldığında hem görsel paneli günceller hem de cümleye ekler
                st.session_state.gorsel_harf = h
                st.session_state.gorsel_guven = 1.0  # Klavyeden basıldığı için %100 güven
                st.session_state.cumle_hafizasi += h
                st.rerun()
