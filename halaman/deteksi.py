# =====================================================
# HALAMAN DETEKSI KEMIRIPAN WAJAH
# =====================================================
# Dikerjakan oleh: [Nama Anggota 4]
# =====================================================

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import time

def tampilkan():
    st.markdown('<h1 class="main-title">🔍 Deteksi Kemiripan Wajah</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Bandingkan dua wajah dengan metode Eigenfaces (PCA)</p>', unsafe_allow_html=True)
    
    # SIDEBAR untuk upload data latih
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="sakura-btn-container">', unsafe_allow_html=True)
        kol1, kol2, kol3 = st.columns([1, 2, 1])
        with kol2:
            if st.button("🌸", key="toggle_sidebar_deteksi"):
                st.session_state.show_upload = not st.session_state.show_upload
                st.rerun()
            st.caption("Klik Sakura")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        if st.session_state.show_upload:
            st.header("📂 Upload Data Latih")
            st.markdown("Upload **minimal 10 foto** wajah (2 orang, masing-masing 5+ foto)")
            
            file_latih = st.file_uploader(
                "Pilih Foto",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key="deteksi_train"
            )
            
            if file_latih:
                st.success(f"✅ {len(file_latih)} foto berhasil terupload!")
            else:
                st.warning("⬆️ Upload foto di sini")
        else:
            st.info("🌸 Upload disembunyikan. Klik sakura di atas.")
        
        st.divider()
        
        ambang = st.slider("🎯 Ambang Batas Kemiripan", 0.0, 1.0, 0.70, 0.05, key="threshold_deteksi")
        st.caption(f"Threshold: {ambang:.2f}")
        
        st.divider()
        
        st.markdown("""
            <b>🌸 Kelompok 2</b><br>
            1. Gea Destadia Al-Zahra<br>
            2. Luna Amilia<br>
            3. Dalilah Arifah Ariandi DJR<br>
            4. Nadia Azzizah
        """, unsafe_allow_html=True)
    
    # AREA UTAMA: Upload 2 foto uji
    st.markdown("## 🔍 Upload Dua Wajah untuk Dibandingkan")
    
    kolom1, kolom2 = st.columns(2)
    with kolom1:
        st.markdown("### 📸 Foto Pertama")
        file1 = st.file_uploader("Upload Foto 1", type=["jpg","jpeg","png"], key="f1_deteksi", label_visibility="collapsed")
    with kolom2:
        st.markdown("### 📸 Foto Kedua")
        file2 = st.file_uploader("Upload Foto 2", type=["jpg","jpeg","png"], key="f2_deteksi", label_visibility="collapsed")
    
    if st.button("🚀 Proses Deteksi Sekarang", use_container_width=True):
        # Ambil file_latih dari session_state (karena di dalam fungsi)
        try:
            train_files = file_latih
        except NameError:
            train_files = None
        
        if not train_files or len(train_files) < 10:
            st.error("⚠️ **Data Latih Kurang!** Upload minimal 10 foto.")
            st.info("💡 Klik tombol 🌸 di sidebar untuk menampilkan bagian upload.")
        elif not file1 or not file2:
            st.error("⚠️ Upload kedua foto uji!")
        else:
            with st.spinner("⏳ Sedang memproses... Mohon tunggu."):
                time.sleep(0.5)
                
                def deteksi_dan_potong_wajah(byte_gambar):
                    arr_np = np.frombuffer(byte_gambar, np.uint8)
                    img = cv2.imdecode(arr_np, cv2.IMREAD_COLOR)
                    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cascade_wajah = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    wajah = cascade_wajah.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    if len(wajah) > 0:
                        x, y, w, h = max(wajah, key=lambda rect: rect[2] * rect[3])
                        potongan = abu[y:y+h, x:x+w]
                        return potongan, True
                    else:
                        return abu, False
                
                def praproses_dengan_deteksi_wajah(byte_gambar, ukuran=(100, 100)):
                    potongan, terdeteksi = deteksi_dan_potong_wajah(byte_gambar)
                    resize = cv2.resize(potongan, ukuran)
                    normal = resize / 255.0
                    vektor = normal.flatten()
                    return vektor, resize, terdeteksi
                
                def muat_gambar_berwarna(byte_gambar, ukuran=(100, 100)):
                    arr_np = np.frombuffer(byte_gambar, np.uint8)
                    img = cv2.imdecode(arr_np, cv2.IMREAD_COLOR)
                    abu = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    cascade_wajah = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    wajah = cascade_wajah.detectMultiScale(abu, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                    if len(wajah) > 0:
                        x, y, w, h = max(wajah, key=lambda rect: rect[2] * rect[3])
                        potongan = img[y:y+h, x:x+w]
                        resize = cv2.resize(potongan, ukuran)
                        return cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                    else:
                        resize = cv2.resize(img, ukuran)
                        return cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)
                
                UKURAN = (100, 100)
                X_latih = []
                
                progress = st.progress(0, text="Mengolah data latih...")
                for i, file in enumerate(train_files):
                    vektor, _, _ = praproses_dengan_deteksi_wajah(file.getvalue(), UKURAN)
                    X_latih.append(vektor)
                    progress.progress((i+1)/len(train_files))
                
                X_latih = np.array(X_latih)
                
                progress.progress(50, text="Menjalankan PCA & mencari Eigenfaces...")
                k = min(50, len(X_latih)-1) if len(X_latih)>1 else 1
                pca = PCA(n_components=k)
                X_pca = pca.fit_transform(X_latih)
                
                progress.progress(70, text="Memproses foto uji...")
                
                vektor1, _, _ = praproses_dengan_deteksi_wajah(file1.getvalue(), UKURAN)
                vektor2, _, _ = praproses_dengan_deteksi_wajah(file2.getvalue(), UKURAN)
                
                gambar1 = muat_gambar_berwarna(file1.getvalue(), UKURAN)
                gambar2 = muat_gambar_berwarna(file2.getvalue(), UKURAN)
                
                proyeksi1 = pca.transform([vektor1])
                proyeksi2 = pca.transform([vektor2])
                
                kemiripan = cosine_similarity(proyeksi1, proyeksi2)[0][0]
                
                progress.progress(100, text="Selesai!")
                time.sleep(0.3)
                progress.empty()
                
                # ===== TAMPILKAN HASIL =====
                st.markdown("---")
                st.subheader("📊 Hasil Deteksi")
                
                kolom_r1, kolom_r2, kolom_r3 = st.columns([2, 2, 1.5])
                
                with kolom_r1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Pertama</div>', unsafe_allow_html=True)
                    st.image(gambar1, caption=f"Resize {UKURAN[0]}x{UKURAN[1]}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with kolom_r2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Kedua</div>', unsafe_allow_html=True)
                    st.image(gambar2, caption=f"Resize {UKURAN[0]}x{UKURAN[1]}", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with kolom_r3:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">🎯 Skor Kemiripan</div>', unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:#AD1457;font-size:42px;'>{kemiripan:.2%}</h1>", unsafe_allow_html=True)
                    
                    if kemiripan >= ambang:
                        st.success("✅ **MIRIP**")
                        st.balloons()
                    elif kemiripan >= 0.50:
                        st.warning("⚠️ **CUKUP MIRIP**")
                    else:
                        st.error("❌ **TIDAK MIRIP**")
                    
                    st.caption(f"Komponen PCA: {k}")
                    st.caption(f"Varians: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
                    st.caption(f"Threshold: {ambang:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # ===== GRAFIK + PENJELASAN =====
                st.markdown("---")
                kolom_graf, kolom_exp = st.columns([1, 1])
                
                with kolom_graf:
                    st.subheader("📈 Grafik Akumulasi Informasi")
                    varians = np.cumsum(pca.explained_variance_ratio_)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    ax.plot(range(1, len(varians)+1), varians, 'bo-', linewidth=2, markersize=5)
                    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Varians')
                    ax.axhline(y=ambang, color='green', linestyle=':', linewidth=2, label=f'Threshold {ambang:.2f}')
                    ax.set_xlabel('Jumlah Komponen PCA (k)', fontsize=10)
                    ax.set_ylabel('Akumulasi Informasi', fontsize=10)
                    ax.set_title('Kurva Akumulasi Informasi PCA', fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='lower right', fontsize=8)
                    ax.set_ylim(0, 1.05)
                    st.pyplot(fig)
                
                with kolom_exp:
                    st.subheader("📖 Penjelasan Grafik")
                    st.markdown("""
                    <div class="explanation-box">
                    Grafik ini menunjukkan seberapa banyak <b>informasi wajah</b> yang bisa dipertahankan jika kita menggunakan sejumlah komponen PCA (k).
                    
                    <br><br>
                    
                    <b>🔵 Garis biru</b> → kurva akumulasi varians. Semakin tinggi, semakin baik.<br>
                    <b>🔴 Garis merah putus-putus</b> → 95% varians data sudah terwakili.<br>
                    <b>🟢 Garis hijau titik-titik</b> → <b>Threshold</b> (batas kemiripan) yang kamu atur di sidebar.
                    
                    <br><br>
                    
                    <b>💡 Cara baca:</b><br>
                    Dari 10.000 pixel wajah, PCA bisa meringkasnya menjadi 50 angka saja tanpa kehilangan banyak informasi. Semakin tinggi garis biru, semakin baik representasi wajahnya.
                    </div>
                    """, unsafe_allow_html=True)