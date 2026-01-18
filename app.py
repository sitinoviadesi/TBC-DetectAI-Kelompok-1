from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm
# PENTING: Tambahkan ImageStat untuk analisis warna
from PIL import Image, ImageStat 
from fpdf import FPDF
from datetime import datetime

# --- KONFIGURASI APP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia-super-aman'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Buat folder upload jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- DATABASE SETUP ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Silakan login untuk mengakses halaman ini."
login_manager.login_message_category = "warning"

# --- MODEL DATABASE ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    riwayats = db.relationship('Riwayat', backref='dokter', lazy=True)

class Riwayat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama_pasien = db.Column(db.String(100), nullable=False)
    umur = db.Column(db.Integer, nullable=False)
    tanggal = db.Column(db.String(50), nullable=False)
    hasil = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.String(20), nullable=False)
    gambar = db.Column(db.String(200), nullable=False)
    risk_level = db.Column(db.String(20), nullable=True)
    recommendation = db.Column(db.Text, nullable=True)
    medical_notes = db.Column(db.Text, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- LOAD AI MODEL ---
try:
    model = load_model('tbc_model.h5')
    print("✅ Model AI Berhasil Dimuat!")
except:
    print("❌ Model tbc_model.h5 TIDAK DITEMUKAN!")
    model = None

# --- FUNGSI VALIDASI GAMBAR (FILTER BARU: HSV SATURATION) ---
def cek_validitas_xray(img_path):
    """
    Mengecek apakah gambar valid menggunakan metode HSV.
    Gambar jalanan/pemandangan punya Saturasi (S) tinggi.
    X-Ray punya Saturasi (S) sangat rendah (hampir 0).
    """
    try:
        # 1. Buka gambar dan ubah ke mode HSV (Hue, Saturation, Value)
        img = Image.open(img_path).convert('HSV')
        
        # 2. Ambil data channel
        h, s, v = img.split()
        
        # 3. Analisis Saturasi (Tingkat Kepekatan Warna)
        stat_s = ImageStat.Stat(s)
        avg_saturation = stat_s.mean[0] # Rata-rata saturasi
        
        # Threshold: X-ray biasanya < 10. Foto biasa > 30.
        # Kita pasang batas 25 untuk toleransi sedikit.
        if avg_saturation > 25: 
            print(f"❌ DITOLAK: Saturasi Tinggi ({avg_saturation:.2f}) - Terdeteksi Gambar Berwarna")
            return False 

        # 4. Analisis Kecerahan (Brightness/Value)
        # Mencegah gambar hitam total atau putih total
        stat_v = ImageStat.Stat(v)
        avg_brightness = stat_v.mean[0]
        
        if avg_brightness < 20: # Terlalu Gelap
            print(f"❌ DITOLAK: Terlalu Gelap ({avg_brightness:.2f})")
            return False
            
        if avg_brightness > 245: # Terlalu Putih
            print(f"❌ DITOLAK: Terlalu Terang ({avg_brightness:.2f})")
            return False

        # Jika lolos semua cek
        print(f"✅ DITERIMA: Gambar Valid (Saturasi: {avg_saturation:.2f})")
        return True

    except Exception as e:
        print(f"Error Validasi: {e}")
        return False

# --- FUNGSI BANTUAN LAIN (GradCAM & PDF) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="out_relu"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)
    superimposed_img.save(output_path)

def get_risk_level_and_details(score):
    """Generate risk level, recommendations, and medical notes based on AI score"""
    if score > 0.8:
        risk_level = "SANGAT TINGGI"
        recommendation = "SEGERA konsultasi dengan dokter spesialis paru (Pulmonologis). Pemeriksaan lanjutan diperlukan: CT scan, tes Mantoux, dan pemeriksaan dahak."
        medical_notes = "Citra radiografi menunjukkan indikasi kuat kelainan paru-paru yang konsisten dengan tuberkulosis. Risiko penularan tinggi jika positif TBC."
    elif score > 0.6:
        risk_level = "TINGGI"
        recommendation = "Segera buat janji temu dengan dokter paru dalam 1-2 hari kerja. Lakukan pemeriksaan laboratorium tambahan (tes tuberkulin, sputum smear microscopy)."
        medical_notes = "Citra menunjukkan temuan yang mencurigakan TBC. Diperlukan konfirmasi medis lebih lanjut untuk diagnosis yang akurat."
    elif score > 0.5:
        risk_level = "SEDANG"
        recommendation = "Konsultasikan hasil dengan dokter untuk evaluasi klinis lebih lanjut. Pertimbangkan pemeriksaan diagnostik tambahan."
        medical_notes = "Temuan borderline. Gejala klinis pasien dan pemeriksaan fisik sangat penting untuk diagnosis final."
    else:
        risk_level = "RENDAH"
        recommendation = "Hasil normal. Tidak ada bukti radiologis tuberkulosis. Namun, tetap ikuti protokol kesehatan standar untuk pencegahan."
        medical_notes = "Citra paru-paru menunjukkan hasil normal. Tidak ada kelainan yang terdeteksi oleh sistem AI."
    
    return risk_level, recommendation, medical_notes

def create_pdf_report(nama, umur, hasil, conf, img_orig, img_heat, risk_level, recommendation, medical_notes):
    """Create a detailed medical PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 12, 'LAPORAN HASIL ANALISIS RADIOGRAFI PARU', 0, 1, 'C')
    pdf.set_font("Arial", 'I', 9)
    pdf.cell(0, 8, 'Sistem Diagnosa Berbasis AI - Deteksi Tuberkulosis Paru', 0, 1, 'C')
    pdf.line(10, 28, 200, 28)
    pdf.ln(5)
    
    # Patient Information
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, 'INFORMASI PASIEN', 0, 1, 'L')
    pdf.set_font("Arial", '', 10)
    pdf.cell(40, 7, 'Nama Pasien')
    pdf.cell(0, 7, f': {nama}', 0, 1)
    pdf.cell(40, 7, 'Usia')
    pdf.cell(0, 7, f': {umur} tahun', 0, 1)
    pdf.cell(40, 7, 'Tanggal Pemeriksaan')
    pdf.cell(0, 7, f': {datetime.now().strftime("%d-%m-%Y %H:%M")}', 0, 1)
    pdf.ln(3)
    
    # Image Section
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, 'CITRA RADIOGRAFI', 0, 1, 'L')
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(60, 5, 'Citra Asli', 0, 0)
    pdf.cell(0, 5, 'Analisis AI (Heatmap)', 0, 1)
    pdf.image(img_orig, x=15, y=pdf.get_y(), w=50)
    pdf.image(img_heat, x=120, y=pdf.get_y()-5, w=50)
    pdf.ln(58)
    pdf.ln(5)
    
    # Diagnosis Results
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, 'HASIL ANALISIS', 0, 1, 'L')
    pdf.set_font("Arial", '', 10)
    
    # Diagnosis status with color indicator
    status_text = f"Status Diagnosis: {hasil}"
    pdf.cell(0, 7, status_text, 0, 1)
    pdf.cell(0, 7, f"Tingkat Keyakinan: {conf}", 0, 1)
    pdf.cell(0, 7, f"Tingkat Risiko: {risk_level}", 0, 1)
    pdf.ln(3)
    
    # Medical Assessment
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, 'PENILAIAN MEDIS', 0, 1, 'L')
    pdf.set_font("Arial", '', 9)
    pdf.multi_cell(0, 5, f"Temuan: {medical_notes}")
    pdf.ln(3)
    
    # Recommendations
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, 'REKOMENDASI TINDAK LANJUT', 0, 1, 'L')
    pdf.set_font("Arial", '', 9)
    pdf.multi_cell(0, 5, f"{recommendation}")
    pdf.ln(5)
    
    # Important Notes
    pdf.set_font("Arial", 'B', 10)
    pdf.set_text_color(180, 0, 0)
    pdf.cell(0, 7, 'PERHATIAN PENTING', 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", '', 8)
    notes = ("1. Laporan ini merupakan hasil analisis AI dan BUKAN DIAGNOSIS MEDIS FINAL\n"
             "2. Diagnosis definitif hanya dapat diberikan oleh dokter spesialis yang berkualifikasi\n"
             "3. Hasil ini harus diinterpretasikan bersama dengan data klinis dan pemeriksaan fisik pasien\n"
             "4. Untuk TBC yang terindikasi, diperlukan tes konfirmasi laboratorium (Mantoux, sputum smear, atau GeneXpert)\n"
             "5. Konsultasikan hasil ini kepada dokter atau rumah sakit terdekat untuk penanganan lebih lanjut")
    pdf.multi_cell(0, 4, notes)
    
    pdf.ln(3)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, f"Laporan dibuat: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}", 0, 1, 'R')
    
    filename = f"Laporan_{nama}_{datetime.now().strftime('%d%m%Y_%H%M%S')}.pdf"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pdf.output(filepath)
    return filename

# --- ROUTES ---

@app.route('/')
def home():
    # Landing Page
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Gagal. Cek email dan password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar.', 'warning')
            return redirect(url_for('register'))
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Berhasil daftar! Silakan login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    data_riwayat = Riwayat.query.filter_by(user_id=current_user.id).order_by(Riwayat.id.desc()).all()
    return render_template('dashboard.html', name=current_user.username, riwayat=data_riwayat)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Anda telah berhasil logout.', 'info') 
    return redirect(url_for('home')) 

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files: return redirect(request.url)
    file = request.files['file']
    nama = request.form.get('nama', 'Pasien')
    umur = request.form.get('umur', '-')
    
    if file.filename == '': return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- VALIDASI GAMBAR (FILTER BARU) ---
        if not cek_validitas_xray(filepath):
            # Jika bukan X-Ray, Hapus file dan Tolak
            os.remove(filepath)
            flash('ERROR: Gambar DITOLAK. Sistem mendeteksi ini bukan Citra X-Ray Paru-Paru (Terlalu berwarna/tidak valid).', 'danger')
            return redirect(url_for('dashboard'))
        # -------------------------------------

        # --- PROSES AI ---
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        score = prediction[0][0]
        
        if score > 0.5:
            hasil = "TERINDIKASI TBC"
            conf = f"{score * 100:.2f}%"
            warna = "danger"
        else:
            hasil = "NORMAL"
            conf = f"{(1 - score) * 100:.2f}%"
            warna = "success"

        # Get risk level and detailed information
        risk_level, recommendation, medical_notes = get_risk_level_and_details(score)

        # Buat Heatmap
        heatmap = make_gradcam_heatmap(img_array, model)
        heatmap_filename = "heatmap_" + filename
        heatmap_path = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
        save_gradcam(filepath, heatmap, heatmap_path)
        
        # Buat PDF dengan informasi detail
        pdf_name = create_pdf_report(nama, umur, hasil, conf, filepath, heatmap_path, risk_level, recommendation, medical_notes)

        # Simpan ke Database
        tgl_sekarang = datetime.now().strftime("%d-%m-%Y %H:%M")
        new_riwayat = Riwayat(
            nama_pasien=nama,
            umur=int(umur),
            tanggal=tgl_sekarang,
            hasil=hasil,
            confidence=conf,
            gambar=filename,
            risk_level=risk_level,
            recommendation=recommendation,
            medical_notes=medical_notes,
            user_id=current_user.id
        )
        db.session.add(new_riwayat)
        db.session.commit()

        return render_template('result.html', nama=nama, hasil=hasil, conf=conf, warna=warna, img_orig=filename, img_heat=heatmap_filename, pdf_link=pdf_name, risk_level=risk_level, recommendation=recommendation, medical_notes=medical_notes)

@app.route('/download/<filename>')
def download_pdf(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

@app.route('/delete_history/<int:id>')
@login_required
def delete_history(id):
    item = Riwayat.query.get_or_404(id)
    if item.user_id == current_user.id:
        db.session.delete(item)
        db.session.commit()
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)