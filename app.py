import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Eğittiğin modeli ve etiketleri yüklüyoruz
model = tf.keras.models.load_model("keras_model.h5") # veya tflite model yükleme yapısı
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_image(img):
    # Görüntüyü modelin beklediği 224x224 boyutuna getir
    img = img.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Tahmin yap
    prediction = model.predict(img_array)
    index = np.argmax(prediction[0])
    confidence = float(prediction[0][index])
    class_name = class_names[index]
    
    return f"Tahmin: {class_name} (Güven: %{confidence * 100:.2f})"

# Gradio arayüzü (Kamera veya Yükleme sekmesi)
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="İşaret Dili Tanıma Asistanı (Teachable Machine + Gradio)"
)

if __name__ == "__main__":
    demo.launch()
