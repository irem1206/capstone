import gradio as gr
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("keras_model.h5", compile=False)
sinif_isimleri = open("labels.txt", "r", encoding="utf-8").readlines()

def isaret_dili_tahmin(image):
    if image is None:
        return "Görüntü bekleniyor..."

    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
    image_array = (image_array / 127.5) - 1
    
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    sinif_adi = sinif_isimleri[index].strip()
    guven_skoru = prediction[0][index]
    
    if guven_skoru > 0.80:
        return f"{sinif_adi} (%{int(guven_skoru * 100)})"
    else:
        return "Anlaşılmadı, tekrar yapın."

arayuz = gr.Interface(
    fn=isaret_dili_tahmin,
    inputs=gr.Image(sources=["webcam"], streaming=True),
    outputs=gr.Textbox(label="Anlık Çeviri", text_align="center"),
    title="Engelsiz Banko: İşaret Dili Çevirmeni",
    description="Kameraya doğru temel işaret dili hareketlerini yapın.",
    live=True
)

arayuz.launch()
