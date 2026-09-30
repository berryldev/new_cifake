import os
import io
import base64
from PIL import Image
import gradio as gr
import uvicorn

# Menggunakan endpoint FastAPI yang sudah ada
from main import app as fastapi_app
from model_loader import predict_image, get_model
from gradcam import generate_gradcam_overlay_base64

def gradio_predict(img):
    if img is None:
        return "Silakan unggah citra terlebih dahulu.", {}, None
    
    # Konversi PIL Image ke bytes
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    img_bytes = buf.getvalue()

    # Jalankan prediksi dengan model asli tanpa perubahan
    result = predict_image(img_bytes)
    label = result["label"]
    confidence = result["confidence"]
    raw_score = result["raw_score"]
    latency = result["inference_time_ms"]

    summary = (
        f"### Prediksi: **{label}**\n"
        f"- **Confidence**: {confidence * 100:.2f}%\n"
        f"- **Raw Score**: {raw_score:.4f}\n"
        f"- **Inference Time**: {latency:.2f} ms\n"
        f"- **Status Model**: {'Simulasi' if result['is_simulated'] else 'Model Aktif'}"
    )

    probs = {
        "REAL": float(raw_score),
        "FAKE": float(1.0 - raw_score)
    }

    # Generate heatmap Grad-CAM++
    model = get_model()
    try:
        overlay_b64 = generate_gradcam_overlay_base64(img_bytes, model=model)
        overlay_bytes = base64.b64decode(overlay_b64)
        overlay_img = Image.open(io.BytesIO(overlay_bytes))
    except Exception as e:
        print(f"[Grad-CAM Error]: {e}")
        overlay_img = None

    return summary, probs, overlay_img

# Buat UI Gradio Interaktif
with gr.Blocks(title="CIFAKE AI Image Detection") as demo:
    gr.Markdown("# 🔍 CIFAKE AI Image Detection API & Web UI")
    gr.Markdown(
        "Deteksi citra sintetis buatan AI vs citra asli (Real) menggunakan model "
        "**EfficientNetB0 + SE-Attention** dengan Explainable AI (**Grad-CAM++**)."
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Unggah Citra (JPG / PNG / WEBP)")
            submit_btn = gr.Button("Analisis Citra", variant="primary")
        with gr.Column():
            output_text = gr.Markdown(label="Ringkasan Hasil")
            output_label = gr.Label(label="Probabilitas Kelas")
            output_cam = gr.Image(type="pil", label="Visualisasi Grad-CAM++ Heatmap")
            
    submit_btn.click(
        fn=gradio_predict,
        inputs=[input_image],
        outputs=[output_text, output_label, output_cam]
    )

    gr.Markdown("""
    ---
    ### 📡 REST API Documentation:
    Space ini sekaligus bertindak sebagai REST API:
    - **`POST /predict`** : Prediksi citra via multipart form-data `file` (JSON response).
    - **`POST /explain`** : Prediksi + Grad-CAM++ Base64 via multipart form-data `file` (JSON response).
    - **`GET /health`** : Cek kesiapan model & server.
    - **`GET /docs`** : Interactive Swagger UI API Documentation.
    """)

# Mount Gradio ke FastAPI pada root "/"
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
