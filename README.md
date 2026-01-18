# 🫁 TBC Detect Pro - AI-Powered Tuberculosis Screening System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> *"Program ini kami dedikasikan untuk mendukung Indonesia Bebas TBC melalui inovasi teknologi yang bermanfaat bagi masyarakat."*

---

## 📑 Daftar Isi
1. [Tentang Proyek](#-tentang-proyek)
2. [Fitur Unggulan](#-fitur-unggulan)
3. [Arsitektur & Teknologi](#-arsitektur--teknologi)
4. [Dataset](#-dataset)
5. [Alur Pengolahan Data](#-alur-pengolahan-data)
6. [Instalasi & Menjalankan](#-instalasi--menjalankan)
7. [Struktur Folder](#-struktur-folder)
8. [Tim Pengembang](#-tim-pengembang)

---

## 📖 Tentang Proyek
**TBC Detect Pro** adalah sistem pakar berbasis web yang memanfaatkan kecerdasan buatan (*Deep Learning*) untuk mendeteksi indikasi Tuberkulosis dari citra X-Ray dada.

Sistem ini dirancang untuk membantu tenaga medis melakukan skrining awal dengan cepat, akurat, dan dilengkapi visualisasi area infeksi menggunakan metode **Grad-CAM (Explainable AI)** untuk transparansi diagnosis.

---

## ✨ Fitur Unggulan

### 🤖 AI Diagnosis & Heatmap
* **Model Cerdas:** Menggunakan arsitektur **MobileNetV2** (Transfer Learning) yang ringan dan akurat.
* **Grad-CAM Visualization:** Menampilkan area paru-paru yang dicurigai terinfeksi dengan peta panas (*heatmap*) berwarna merah.

### 🛡️ Smart Image Filter (Anti-Spam)
* Sistem otomatis menolak gambar yang bukan X-Ray (seperti foto selfie, pemandangan, atau objek acak).
* Menggunakan algoritma **HSV Saturation Check** & **Brightness Threshold** untuk memastikan validitas citra medis.

### ⚡ UX Modern
* **Loading Animation:** Indikator visual interaktif saat AI sedang memproses data.
* **Dashboard Responsif:** Tampilan rapi dan fleksibel di Desktop maupun Mobile.

### 📂 Manajemen Data Lengkap
* **PDF Reporting:** Cetak hasil diagnosis medis resmi secara otomatis dalam format PDF.
* **Riwayat & Pencarian:** Penyimpanan data pasien terpusat dengan fitur pencarian cepat.
* **Keamanan:** Password pengguna dienkripsi menggunakan standar **SHA-256**.

---

## 🛠 Arsitektur & Teknologi

Aplikasi ini dibangun menggunakan ekosistem Python yang handal:

* **Backend:** Flask (Python Microframework).
* **AI Core:** TensorFlow & Keras (MobileNetV2).
* **Image Processing:**
    * **NumPy:** Komputasi matriks gambar.
    * **Pillow (PIL):** Validasi warna dan manipulasi citra.
    * **Matplotlib:** Pembuatan visualisasi *heatmap*.
* **Database:** SQLAlchemy (SQLite) untuk penyimpanan data user & riwayat pasien.
* **Security:** Werkzeug Security (Password Hashing).

---

## 📂 Dataset

Penelitian ini menggunakan dataset sekunder yang diperoleh dari repositori publik Kaggle.

🔗 **Sumber Dataset:** [Tuberculosis (TB) Chest X-ray Database - Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-database)

**Rincian Data:**
* **Total Data:** 4.200 Citra
* **Kelas Normal:** 3.500 Citra
* **Kelas Tuberculosis:** 700 Citra
* **Format Citra:** .png / .jpg

---

## 🔄 Alur Pengolahan Data

Sebelum diagnosis ditampilkan, gambar melewati tahapan ketat berikut:

1.  **Input:** User mengupload file (`.jpg`, `.png`).
2.  **Validasi Otomatis:**
    * *Cek Saturasi Warna (HSV):* Jika gambar terlalu berwarna (Saturasi > 25), otomatis **DITOLAK**.
    * *Cek Kecerahan:* Jika terlalu gelap atau terlalu terang, otomatis **DITOLAK**.
3.  **Preprocessing:**
    * Resize ke dimensi **224x224 pixel**.
    * Normalisasi pixel (Rescaling 1./255).
4.  **Inference (AI):** Model MobileNetV2 memprediksi probabilitas TBC.
5.  **Post-Processing:**
    * Generate Heatmap Grad-CAM.
    * Simpan hasil ke Database.
    * Generate laporan PDF siap unduh.

---

## 🚀 Instalasi & Menjalankan

Ikuti langkah-langkah berikut untuk menjalankan proyek di komputer lokal (Localhost).

### 1. Clone Repository
```bash
git clone [https://github.com/yesaprayugo/TBC-DetectAI-Kelompok-1.git](https://github.com/yesaprayugo/TBC-DetectAI-Kelompok-1.git)
cd TBC-DetectAI-Kelompok-1
