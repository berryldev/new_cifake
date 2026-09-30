---
title: CIFAKE AI Image Detection
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# CIFAKE AI Image Detection API & Web UI (100% Gratis di Hugging Face Spaces)

Layanan klasifikasi citra sintetis buatan kecerdasan buatan (AI-Generated / Sintetis) vs citra asli/nyata menggunakan model deep learning **EfficientNetB0 + SE-Attention**, dilengkapi dengan visualisasi Explainable AI (**Grad-CAM++**).

> **Catatan Biaya**: Hugging Face mewajibkan akun berbayar (Pro) untuk tipe Space *Docker*. Oleh karena itu, repository ini dikonfigurasi menggunakan SDK **Gradio (Gratis)** yang secara cerdas me-mount backend **FastAPI**, sehingga Anda mendapatkan **REST API lengkap + Web UI interaktif** secara **100% GRATIS (CPU Basic 16 GB RAM)** tanpa perlu kartu kredit!

---

## 🚀 Panduan Upload ke Hugging Face Spaces (100% Gratis)

### Langkah 1: Buat Space Baru di Hugging Face
1. Buka [Hugging Face Spaces](https://huggingface.co/spaces) dan login ke akun Anda.
2. Klik **Create new Space**.
3. Isi informasi Space:
   - **Space name**: misalnya `cifake-detection-api`
   - **License**: `mit` (atau pilihan Anda)
   - **Select the Space SDK**: Pilih **Gradio** ⭐ *(Bukan Docker, agar 100% Gratis)*
   - **Space hardware**: `CPU Basic (2 vCPU, 16 GB RAM) - Free`
4. Klik **Create Space**.

---

### Langkah 2: Upload File ke Space

Pilih salah satu dari dua cara berikut:

#### Opsi A: Menggunakan Git & Git LFS (Direkomendasikan)
1. Buka terminal di folder ini:
   ```bash
   git remote add origin https://huggingface.co/spaces/<USERNAME>/<NAMA_SPACE>
   git push -u origin main
   ```
   *(Ganti `<USERNAME>` dan `<NAMA_SPACE>` sesuai akun Anda. Masukkan token akses Hugging Face role `Write` jika diminta password)*.

#### Opsi B: Upload Langsung via Web Hugging Face
1. Masuk ke halaman Space yang Anda buat di browser.
2. Klik tab **Files and versions** $\rightarrow$ **Add file** $\rightarrow$ **Upload files**.
3. Upload seluruh file utama berikut:
   - `cifake_model.keras`
   - `app.py`
   - `main.py`
   - `model_loader.py`
   - `gradcam.py`
   - `requirements.txt`
   - `README.md`
   - `.gitattributes`
4. Klik **Commit changes to main**.

---

### Langkah 3: Tunggu Proses Build Selesai
Hugging Face akan otomatis memasang library di `requirements.txt` dan menjalankan `app.py`. Setelah statusnya **Running**, aplikasi dan API Anda langsung aktif secara gratis!

---

## 📡 Dokumentasi Endpoint REST API

Semua endpoint FastAPI tetap dapat diakses secara langsung:

| Endpoint | Method | Keterangan |
|---|---|---|
| `/` | `GET` | Web UI Interaktif (Gradio) untuk uji coba langsung |
| `/docs` | `GET` | Dokumentasi Interaktif Swagger UI FastAPI |
| `/health` | `GET` | Cek status server dan model |
| `/predict` | `POST` | Prediksi klasifikasi REAL / FAKE (JSON) |
| `/explain` | `POST` | Prediksi klasifikasi + Heatmap Grad-CAM++ Base64 (JSON) |

### Contoh Pemanggilan via Python (`requests`):
```python
import requests

HF_API_URL = "https://<USERNAME>-<NAMA_SPACE>.hf.space"

# 1. Prediksi biasa
with open("gambar_uji.jpg", "rb") as f:
    res = requests.post(f"{HF_API_URL}/predict", files={"file": f}).json()
print("Hasil:", res)

# 2. Prediksi + Grad-CAM++
with open("gambar_uji.jpg", "rb") as f:
    res_xai = requests.post(f"{HF_API_URL}/explain", files={"file": f}).json()
print("Grad-CAM Base64:", res_xai.get("gradcam_overlay_base64")[:50])
```

### Contoh Pemanggilan via cURL:
```bash
curl -X POST "https://<USERNAME>-<NAMA_SPACE>.hf.space/predict" \
  -F "file=@gambar_uji.jpg"
```
