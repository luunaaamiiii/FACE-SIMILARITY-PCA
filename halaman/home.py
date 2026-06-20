# =====================================================
# HALAMAN HOME - BERANDA & PROFIL
# =====================================================
# Dikerjakan oleh: [Nama Anggota 1]
# =====================================================

import streamlit as st

def tampilkan():
    st.markdown('<h1 class="main-title">🌸 Selamat Datang di Aplikasi PCA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Kelompok 2 – Aljabar Linier / Computer Vision</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class="explanation-box">
        <h3>📌 Tentang Aplikasi</h3>
        Aplikasi ini dibuat untuk memenuhi tugas mata kuliah Aljabar Linier / Computer Vision.
        Kami mengimplementasikan metode <b>PCA (Principal Component Analysis)</b> untuk:
        <ul>
            <li><b>🌫️ Grayscale</b> – Mengubah gambar berwarna menjadi hitam-putih</li>
            <li><b>🗜️ Kompresi</b> – Mengompresi gambar dengan PCA, menampilkan SSIM, PSNR, dan rasio kompresi</li>
            <li><b>🔍 Deteksi Kemiripan</b> – Membandingkan dua wajah menggunakan Eigenfaces</li>
        </ul>
        Gunakan menu emoji di samping untuk memilih fitur.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.smarkdown("""
        <div class="result-card">
        <h4>👥 Anggota Kelompok</h4>
        <p>1. Gea Destadia Al-Zahra<br>
        2. Luna Amilia<br>
        3. Dalilah Arifah Ariandi DJR<br>
        4. Nadia Azzizah</p>
        <p style="font-size:14px; color:#AD1457;">🌸 Terima kasih telah berkunjung!</p>
        </div>
        """, unsafe_allow_html=True)