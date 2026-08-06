import streamlit as st
import numpy as np
import cv2
import requests
import base64
from PIL import Image

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="İşaret Dili Asistanı", page_icon="🤟", layout="wide", initial_sidebar_state="collapsed")

# --- MODERN STİLLER ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f3f4f6; }
    .hero-container { background: linear-gradient(135deg, #1f2937 0%, #111827 100%); padding: 30px; border-radius: 15px; border: 1px solid #374151; text-align: center; margin-bottom: 25px; }
    .hero-title { color: #60a5fa; font-size: 2.2em; font-weight: 800; margin: 0; }
    .hero-subtitle { color: #9ca3af; font-size: 1.1em; margin-top: 10px; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; background-color: #2563eb; color: white; border: none; }
    .stButton>button:hover { background-color: #1d4ed8; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='hero-container'>
        <p class='hero-title'>🤟 Yapay Zeka Destekli Türk İşaret Dili Çeviri Asistanı</p>
        <p class='hero-subtitle'>Roboflow RF-DETR Tabanlı Gerçek Zamanlı Görsel Tanıma ve Dinamik Cümle Sentezleme Motoru</p>
    </div>
""", unsafe_allow_html=True)

if 'cumle' not in st.session_state: st.session_state.cumle = ""
if 'aktif_harf' not in st.session_state: st.session_state.aktif_harf = ""
if 'aktif_guven' not in st.session_state: st.session_state.aktif_guven = 0.0

ROBOFLOW_MODEL_ID = "turkish-sign-language-letters/1"
ROBOFLOW_API_KEY = "ggw75nomaYTUJtoijwI4" 

def roboflow_tahmin_yap(frame):
    """Görseli Roboflow Inference API'ye gönderir ve kutu/etiket sonuçlarını döner."""
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "BURAYA_ROBOFLOW_API_KEYINIZI_GIRIN":
        return frame, "API KEY EKSİK", 0.0

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _, img_encoded = cv2.imencode('.jpg', frame_rgb)
        img_bytes = img_encoded.tobytes()

        upload_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}?api_key={ROBOFLOW_API_KEY}&confidence=65"
        
        response = requests.post(
            upload_url,
            data=img_bytes,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        if response.status_code != 200:
            st.error(f"Roboflow API Hatası ({response.status_code}): {response.text}")
            return frame, "API Hatası", 0.0

        predictions = response.json().get("predictions", [])
        
        if not predictions:
            return frame, "Bilinmeyen", 0.0

        processed_frame = frame_rgb.copy()
        best_class = "Bilinmeyen"
        max_conf = 0.0

        # Tespit edilen nesneleri/harfleri çiz
       # Tespit edilen nesneleri/harfleri çiz
        for pred in predictions:
            x, y, w, h = int(pred['x']), int(pred['y']), int(pred['width']), int(pred['height'])
            label = str(pred['class']).upper()
            conf = float(pred['confidence'])

            if conf > max_conf:
                max_conf = conf
                best_class = label

            # Kutu koordinatları
            pt1 = (int(x - w / 2), int(y - h / 2))
            pt2 = (int(x + w / 2), int(y + h / 2))

            # Bounding box & yazı çizimi
            cv2.rectangle(processed_frame, pt1, pt2, (0, 255, 0), 3)
            cv2.putText(
                processed_frame, 
                f"{label} %{int(conf*100)}", 
                (pt1[0], max(pt1[1] - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )

        
        if max_conf < 0.65:
            return processed_frame, "Algılanamadı (TİD Uyumsuz)", max_conf
       

        return processed_frame, best_class, max_conf

    except Exception as e:
        return frame, "Hata", 0.0

        return processed_frame, best_class, max_conf

    except Exception as e:
        return frame, "Hata", 0.0

col1, col2 = st.columns([1.2, 1], gap="large")

out_frame = None

with col1:
    st.subheader("📹 Canlı Veri Akışı")
    girdi = st.radio("Seçim:", ["Kamera", "Fotoğraf"], horizontal=True, label_visibility="collapsed")
    frame = None
    if girdi == "Fotoğraf":
        file = st.file_uploader("Resim seç", type=["jpg", "jpeg", "png"])
        if file:
            frame = cv2.imdecode(np.asarray(bytearray(file.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        cam = st.camera_input("Kamera")
        if cam:
            frame = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)

if frame is not None:
    out_frame, h, c = roboflow_tahmin_yap(frame)
    st.session_state.aktif_harf = h
    st.session_state.aktif_guven = c
    st.image(out_frame, caption="Roboflow Tespit Sonucu", use_column_width=True)

with col2:
    st.subheader("📊 Model Çıkarım Paneli")
    if st.session_state.aktif_harf == "":
        st.info("Kameradan, fotoğraftan veya en aşağıdaki klavyeden bir giriş bekliyorum...")
    else:
        harf = st.session_state.aktif_harf
        guven = st.session_state.aktif_guven
        
        if guven < 0.40 and frame is not None:
            st.markdown(f"<div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #ef4444; text-align: center;'><h3 style='margin:0; color: #ef4444;'>⚠️ Düşük Güven</h3><h1 style='margin:10px 0; color: #f87171;'>{harf}</h1><p style='color: #9ca3af;'>Güven: %{guven*100:.1f}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #1f2937; padding: 25px; border-radius: 12px; border: 1px solid #10b981; text-align: center;'><span style='color: #9ca3af; text-transform: uppercase;'>Aktif Tahmin / Seçim</span><h1 style='margin: 10px 0; color: #34d399; font-size: 5em;'>{harf}</h1><span style='color: #9ca3af;'>Güven Skoru: %{guven*100:.1f}</span></div>", unsafe_allow_html=True)
        st.progress(float(guven))

st.markdown("---")
st.subheader("📝 Metin ve Ses Motoru")

c1, c2 = st.columns(2)
if c1.button("➕ Yukarıdaki Harfi Ekle", type="primary"):
    if st.session_state.aktif_harf and st.session_state.aktif_harf not in ["BİLİNMEYEN", "HATA"]:
        st.session_state.cumle += st.session_state.aktif_harf
        st.rerun()
if c2.button("␣ Boşluk Ekle"):
    st.session_state.cumle += " "
    st.rerun()

st.session_state.cumle = st.text_input("Çıktı:", value=st.session_state.cumle, label_visibility="collapsed")

metin_js = st.session_state.cumle.replace("'", "\\'")
st.markdown(f"<button onclick=\"const u = new SpeechSynthesisUtterance('{metin_js}'); u.lang='tr-TR'; window.speechSynthesis.speak(u);\" style='width: 100%; background-color: #2563eb; color: white; padding: 10px; border: none; border-radius: 8px; font-weight: bold; cursor: pointer;'>🔊 Sesli Oku</button>", unsafe_allow_html=True)

c3, c4 = st.columns(2)
if c3.button("⬅️ Sil"):
    st.session_state.cumle = st.session_state.cumle[:-1]
    st.rerun()
if c4.button("🧹 Temizle"):
    st.session_state.cumle = ""
    st.rerun()

st.markdown("---")
st.subheader("⌨️ Sanal Klavye")

alfabe = [
    ["A","B","C","Ç","D","E","F","G","Ğ"], 
    ["H","I","İ","J","K","L","M","N","O"], 
    ["Ö","P","R","S","Ş","T","U","Ü","V"], 
    ["Y","Z"]
]

for satir in alfabe:
    cols = st.columns(len(satir))
    for i, harf in enumerate(satir):
        if cols[i].button(harf, key=f"k_{harf}"):
            st.session_state.aktif_harf = harf
            st.session_state.aktif_guven = 1.0  
            st.session_state.cumle += harf
            st.rerun()
