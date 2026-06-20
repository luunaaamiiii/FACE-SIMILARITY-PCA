# =====================================================
# HALAMAN KOMPRESI - KOMPRESI GAMBAR DENGAN PCA
# =====================================================
# Dikerjakan oleh: [Nama Anggota 3]
# =====================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ==========================================
# IMPOR SKIMAGE (jika ada)
# ==========================================
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

def tampilkan():
    st.markdown('<h1 class="main-title">🗜️ Kompresi Gambar dengan PCA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload gambar, atur jumlah komponen, lihat hasil kompresi & metrik kualitas</p>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], key="kompresi_upload")
    if file is not None:
        img = Image.open(file).convert("L")
        img_np = np.array(img, dtype=np.float64)
        h, w = img_np.shape
        
        if h > 300 or w > 300:
            st.warning("Gambar terlalu besar, akan di-resize ke 256x256 agar proses cepat")
            img_resized = img.resize((256, 256))
            img_np = np.array(img_resized, dtype=np.float64)
            h, w = img_np.shape
        
        k_max = min(h, w)
        k = st.slider("Jumlah komponen PCA (k)", min_value=1, max_value=k_max, value=min(50, k_max), step=1)
        
        mean_col = np.mean(img_np, axis=0)
        centered = img_np - mean_col
        cov = np.cov(centered, rowvar=False)
        eigen_vals, eigen_vecs = np.linalg.eigh(cov)
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]
        Vk = eigen_vecs[:, :k]
        proj = centered @ Vk
        rekon = proj @ Vk.T + mean_col
        rekon = np.clip(rekon, 0, 255).astype(np.uint8)
        
        img_uint8 = img_np.astype(np.uint8)
        if SKIMAGE_AVAILABLE:
            ssim_val = ssim(img_uint8, rekon, data_range=255)
            psnr_val = psnr(img_uint8, rekon, data_range=255)
        else:
            ssim_val = "Tidak tersedia"
            psnr_val = "Tidak tersedia"
        
        original_size = h * w
        compressed_size = h * k + w * k
        compression_ratio = compressed_size / original_size
        saving = (1 - compression_ratio) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_uint8, caption="Gambar Asli (Grayscale)", use_container_width=True)
        with col2:
            st.image(rekon, caption=f"Hasil Kompresi (k={k})", use_container_width=True)
        
        st.markdown("### 📊 Metrik Kualitas")
        c1, c2, c3 = st.columns(3)
        c1.metric("SSIM", f"{ssim_val:.4f}" if isinstance(ssim_val, float) else ssim_val)
        c2.metric("PSNR", f"{psnr_val:.2f} dB" if isinstance(psnr_val, float) else psnr_val)
        c3.metric("Penghematan", f"{saving:.1f}%")
        
        st.markdown(f"""
        <div class="explanation-box">
        <b>Detail Kompresi:</b><br>
        • Ukuran asli: {original_size} pixel<br>
        • Ukuran setelah PCA (approx): {compressed_size} koefisien<br>
        • Rasio kompresi: {compression_ratio:.4f}<br>
        • Jumlah komponen PCA: {k}
        </div>
        """, unsafe_allow_html=True)
        
        total_var = np.sum(eigen_vals)
        cum_var = np.cumsum(eigen_vals) / total_var
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cum_var)+1), cum_var, 'bo-', linewidth=2)
        ax.axhline(y=0.95, color='r', linestyle='--', label='95% Varians')
        ax.axvline(x=k, color='g', linestyle=':', label=f'k = {k}')
        ax.set_xlabel('Jumlah Komponen (k)')
        ax.set_ylabel('Akumulasi Varians')
        ax.set_title('Kurva Akumulasi Informasi PCA')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)