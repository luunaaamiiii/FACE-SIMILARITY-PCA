elif page == "🔍 Deteksi":
    if not st.session_state.deteksi_visited:
        st.balloons()
        st.session_state.deteksi_visited = True

    st.markdown("""
    <div class="deteksi-header">
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
        <h1>🔍 Deteksi Kemiripan Wajah</h1>
        <p>Bandingkan dua wajah dengan metode PCA (Eigenfaces) dan Cosine Similarity.</p>
        <div class="love-shower">❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖 ❤️ 💖</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: linear-gradient(135deg, #FCE4EC, #FFF0F5); 
                padding: 1.5rem; border-radius: 16px; border: 1px solid #F8BBD0; 
                margin-bottom: 2rem; text-align: center;">
        <p style="font-size:1.2rem; color:#6A1B4D;">
            ❤️ <b>Cara kerja:</b> PCA mengekstrak fitur utama (eigenfaces) dari data latih (wajah). 
            Dua wajah yang dibandingkan diproyeksikan ke ruang PCA, lalu dihitung kemiripannya dengan <b>Cosine Similarity</b>.
            Semakin tinggi skor, semakin mirip kedua wajah.
        </p>
        <p style="color:#880E4F; font-style:italic;">
            "Setiap wajah unik, tapi kecocokan bisa ditemukan."
        </p>
        <p>📌 <b>Keterangan:</b> Deteksi kemiripan menggunakan PCA (Eigenfaces) dan Cosine Similarity. 
        Upload data latih (ZIP) untuk hasil lebih akurat, atau biarkan sistem menggunakan data latih default LFW.</p>
    </div>
    """, unsafe_allow_html=True)

    # ========================================================
    # FUNGSI VALIDASI WAJAH (deteksi apakah gambar mengandung wajah manusia)
    # ========================================================
    def is_face_image(byte_gambar, min_confidence=5):
        """
        Cek apakah gambar mengandung wajah manusia menggunakan Haar Cascade.
        Kembalikan True jika ada wajah, False jika tidak.
        """
        try:
            np_arr = np.frombuffer(byte_gambar, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=min_confidence,  # makin tinggi makin ketat
                minSize=(50, 50)
            )
            return len(faces) > 0
        except Exception as e:
            print(f"Error validasi wajah: {e}")
            return False

    # ========================================================
    # SESSION STATE UNTUK MODEL
    # ========================================================
    if "deteksi_model_loaded" not in st.session_state:
        st.session_state.deteksi_model_loaded = False
        st.session_state.deteksi_pca_model = None
        st.session_state.deteksi_X_train = None

    # ========================================================
    # LOAD DATA LATIH DEFAULT (LFW) HANYA SEKALI
    # ========================================================
    if not st.session_state.deteksi_model_loaded:
        with st.spinner("⏳ Memuat dataset LFW untuk data latih default... Tunggu yaa ^^"):
            try:
                lfw = fetch_lfw_people(min_faces_per_person=5, resize=0.4, color=False)
                unique_labels = np.unique(lfw.target)
                valid_labels = [label for label in unique_labels if np.sum(lfw.target == label) >= 5]
                if len(valid_labels) >= 2:
                    selected = valid_labels[:2]
                    X_train = []
                    for label in selected:
                        idx = np.where(lfw.target == label)[0][:5]
                        for i in idx:
                            img = cv2.resize(lfw.images[i], (100, 100)).flatten() / 255.0
                            X_train.append(img)
                    X_train = np.array(X_train)
                    k = min(50, len(X_train)-1)
                    pca = PCA(n_components=k)
                    pca.fit(X_train)
                    st.session_state.deteksi_pca_model = pca
                    st.session_state.deteksi_X_train = X_train
                    st.session_state.deteksi_model_loaded = True
                    st.success(f"✅ Data latih default LFW dimuat ({len(X_train)} foto dari 2 orang)")
                else:
                    st.warning("⚠️ Dataset LFW tidak mencukupi (minimal 2 orang dengan 5 foto). Upload data latih sendiri.")
            except Exception as e:
                st.warning(f"Gagal memuat LFW: {e}. Upload data latih sendiri.")

    # ========================================================
    # PILIH SUMBER DATA LATIH
    # ========================================================
    st.markdown("---")
    st.markdown("#### 📂 Data Latih")
    data_mode = st.radio(
        "Pilih sumber data latih:",
        ["Gunakan data latih default (LFW)", "Upload file ZIP berisi gambar wajah"],
        horizontal=True,
        key="data_mode"
    )

    uploaded_zip = None
    if data_mode == "Upload file ZIP berisi gambar wajah":
        uploaded_zip = st.file_uploader("Unggah file ZIP", type=["zip"], key="train_zip_deteksi")
        if uploaded_zip is not None:
            st.success("✅ File ZIP berhasil diunggah.")

    # ========================================================
    # UPLOAD DUA FOTO UJI
    # ========================================================
    col_upload1, col_upload2 = st.columns(2)
    with col_upload1:
        img1 = st.file_uploader("📤 Foto Pertama", type=["jpg", "jpeg", "png"], key="img1_deteksi")
    with col_upload2:
        img2 = st.file_uploader("📤 Foto Kedua", type=["jpg", "jpeg", "png"], key="img2_deteksi")

    # ========================================================
    # PARAMETER: KOMPONEN PCA & THRESHOLD
    # ========================================================
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        n_components = st.slider("Jumlah komponen PCA (k)", 2, 50, 9, 1, key="n_comp_deteksi")
    with col_param2:
        # DEFAULT THRESHOLD DINAIKKAN DARI 70% → 80% (LEBIH KETAT)
        threshold = st.slider("Threshold kemiripan (%)", 0, 100, 80, 5, key="thresh_deteksi") / 100.0

    # ========================================================
    # PROSES DETEKSI
    # ========================================================
    if img1 is not None and img2 is not None:
        col_show1, col_show2 = st.columns(2)
        with col_show1:
            st.image(img1, caption="Foto Pertama", use_container_width=True)
        with col_show2:
            st.image(img2, caption="Foto Kedua", use_container_width=True)

        if st.button("🔎 Hitung Kemiripan", use_container_width=True):
            try:
                # ----- VALIDASI WAJAH (deteksi apakah benar-benar wajah manusia) -----
                img1_bytes = img1.getvalue()
                img2_bytes = img2.getvalue()

                if not is_face_image(img1_bytes):
                    st.error("❌ **Foto Pertama BUKAN WAJAH MANUSIA!** Upload foto wajah yang jelas dan tidak menggunakan filter.")
                    st.stop()
                if not is_face_image(img2_bytes):
                    st.error("❌ **Foto Kedua BUKAN WAJAH MANUSIA!** Upload foto wajah yang jelas dan tidak menggunakan filter.")
                    st.stop()

                # ----- PREPROCESSING -----
                size = (100, 100)
                im1 = Image.open(img1).convert("L").resize(size)
                im2 = Image.open(img2).convert("L").resize(size)
                arr1 = np.array(im1, dtype=np.float32).flatten() / 255.0
                arr2 = np.array(im2, dtype=np.float32).flatten() / 255.0

                # ----- TENTUKAN DATA LATIH -----
                train_vectors = None
                if data_mode == "Gunakan data latih default (LFW)" and st.session_state.deteksi_model_loaded:
                    train_vectors = st.session_state.deteksi_X_train
                    pca = st.session_state.deteksi_pca_model
                elif data_mode == "Upload file ZIP berisi gambar wajah" and uploaded_zip is not None:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                            zip_ref.extractall(tmpdir)
                        train_vectors = []
                        for root, _, files in os.walk(tmpdir):
                            for file in files:
                                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    try:
                                        img_path = os.path.join(root, file)
                                        img = Image.open(img_path).convert("L").resize(size)
                                        vec = np.array(img, dtype=np.float32).flatten() / 255.0
                                        train_vectors.append(vec)
                                    except:
                                        continue
                        if len(train_vectors) < 2:
                            st.error("Data latih dari ZIP kurang dari 2 gambar. Gagal melatih PCA.")
                            st.stop()
                        train_vectors = np.array(train_vectors)
                        k = min(n_components, len(train_vectors)-1, len(train_vectors[0]))
                        pca = PCA(n_components=k)
                        pca.fit(train_vectors)
                else:
                    st.error("Tidak ada data latih yang valid. Pilih sumber data latih atau upload ZIP.")
                    st.stop()

                # ----- PROYEKSI KE RUANG PCA -----
                vec1_pca = pca.transform([arr1])[0]
                vec2_pca = pca.transform([arr2])[0]

                # ----- HITUNG COSINE SIMILARITY -----
                sim = cosine_similarity([vec1_pca], [vec2_pca])[0][0]
                kemiripan = sim
                var_ratio = pca.explained_variance_ratio_.sum() * 100
                ambang = threshold

                # ----- TAMPILKAN HASIL (3 KOLOM) -----
                st.subheader("Hasil Deteksi Foto Kamu ^^")
                kolom_r1, kolom_r2, kolom_r3 = st.columns([2, 2, 1.5])
                with kolom_r1:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Pertama</div>', unsafe_allow_html=True)
                    st.image(img1, caption="Foto Asli", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r2:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">📸 Foto Kedua</div>', unsafe_allow_html=True)
                    st.image(img2, caption="Foto Asli", use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with kolom_r3:
                    st.markdown('<div class="result-container">', unsafe_allow_html=True)
                    st.markdown('<div class="pink-badge">Skor Kemiripan Foto!!</div>', unsafe_allow_html=True)
                    st.markdown(f"<h1 style='color:#AD1457;font-size:42px;'>{kemiripan:.2%}</h1>", unsafe_allow_html=True)
                    if kemiripan >= ambang:
                        st.success("**WAH MIRIP!! :D**")
                        st.balloons()
                    elif kemiripan >= 0.50:
                        st.warning("**HMM CUKUP MIRIP LAH YA ;D**")
                    else:
                        st.error("**TIDAK MIRIP ^^**")
                    st.caption(f"Komponen PCA: {pca.n_components_}")
                    st.caption(f"Varians: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                # ----- GRAFIK AKUMULASI INFORMASI -----
                st.markdown("---")
                kolom_graf, kolom_exp = st.columns([1, 1])
                with kolom_graf:
                    st.subheader("Grafik Akumulasi Informasi")
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
                    st.subheader("Penjelasan Grafik!!")
                    st.markdown("""
                    <div class="explanation-box">
                    Grafik ini menunjukkan seberapa banyak <b>informasi wajah</b> yang bisa dipertahankan jika kita menggunakan sejumlah komponen PCA (k).
                    <br><br>
                    <b>🔵 Garis biru</b> → kurva akumulasi varians. Semakin tinggi, semakin baik.<br>
                    <b>🔴 Garis merah putus-putus</b> → 95% varians data sudah terwakili.<br>
                    <b>🟢 Garis hijau titik-titik</b> → <b>Threshold</b> (batas kemiripan) yang kamu atur.
                    <br><br>
                    <b>💡 Cara baca:</b><br>
                    Dari 10.000 pixel wajah, PCA bisa meringkasnya menjadi 50 angka saja tanpa kehilangan banyak informasi. Semakin tinggi garis biru, semakin baik representasi wajahnya.
                    </div>
                    """, unsafe_allow_html=True)

                st.balloons()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    else:
        st.markdown("""
        <div style="text-align:center; padding:2rem 0;">
            <p style="font-size:1.2rem; color:#6A1B4D;">👆 Upload dua foto wajah untuk membandingkan.</p>
        </div>
        """, unsafe_allow_html=True)
