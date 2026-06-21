import streamlit as st
import halaman.home as home
import halaman.grayscale as grayscale
import halaman.kompresi as kompresi
import halaman.deteksi as deteksi
import base64
import os
import requests
from urllib.parse import urlparse

# ======================== KONFIGURASI HALAMAN ========================
st.set_page_config(
    page_title="ANGEL APP",
    page_icon="🌸",
    layout="wide"
)

# ======================== CSS GLOBAL (sama persis dengan kode asli) ========================
st.markdown("""
    <style>
        /* ----- BACKGROUND & WARNA DASAR ----- */
        .stApp, .main, .block-container, section.main, div[data-testid="stSidebar"] {
            background-color: #FFF0F5 !important;
        }
        body, p, div, span, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stCaption {
            color: #6A1B4D !important;
        }
        header {
            background: linear-gradient(135deg, #880E4F, #AD1457) !important;
            border-bottom: 2px solid #F8BBD0 !important;
        }
        header * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }
        .css-1d391kg, .css-12w0qpk, [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #FCE4EC, #FFF0F5) !important;
            border-right: 2px solid #F8BBD0 !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #AD1457 !important;
            font-weight: bold !important;
        }

        /* ----- SIDEBAR DEKORASI ----- */
        .sidebar-header {
            text-align: center;
            padding: 5px 0 5px 0;
            border-bottom: 2px solid #F8BBD0;
            margin-bottom: 10px;
        }
        .sidebar-header .logo {
            font-size: 40px;
            display: block;
            margin-bottom: 2px;
        }
        .sidebar-header .title {
            font-size: 22px;
            font-weight: bold;
            color: #AD1457;
            letter-spacing: 2px;
        }
        .sidebar-header .subtitle {
            font-size: 13px;
            color: #880E4F;
            font-style: italic;
            margin-top: 2px;
        }
        .sidebar-footer {
            text-align: center;
            font-size: 12px;
            color: #AD1457;
            border-top: 1px solid #F8BBD0;
            padding-top: 8px;
            margin-top: 12px;
        }

        /* ----- TOMBOL UMUM ----- */
        .stButton button {
            background: linear-gradient(135deg, #EC407A, #D81B60) !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 50px !important;
            padding: 10px 30px !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(233, 30, 99, 0.3) !important;
            transition: 0.3s !important;
        }
        .stButton button:hover {
            transform: scale(1.03) translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(233, 30, 99, 0.4) !important;
        }

        /* ----- FILE UPLOADER ----- */
        div[data-testid="stFileUploader"] {
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5) !important;
            border: 2px dashed #EC407A !important;
            border-radius: 12px !important;
            padding: 10px !important;
        }
        div[data-testid="stFileUploader"] > div {
            background: rgba(255, 255, 255, 0.6) !important;
            border-radius: 8px !important;
            padding: 20px !important;
        }
        div[data-testid="stFileUploader"] * {
            color: #6A1B4D !important;
            background: transparent !important;
        }
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #EC407A, #D81B60) !important;
            color: white !important;
            border: none !important;
            border-radius: 20px !important;
            padding: 5px 20px !important;
        }
        div[data-testid="stFileUploader"] button:hover {
            transform: scale(1.05) !important;
        }
        div[data-testid="stFileUploader"]:hover {
            border-color: #D81B60 !important;
        }

        /* ----- SLIDER ----- */
        .stSlider > div {
            background: rgba(255, 255, 255, 0.4) !important;
            border-radius: 10px !important;
        }

        /* ----- SIDEBAR NAVIGASI (tombol bulat) ----- */
        .stSidebar .stButton button {
            width: 48px !important;
            height: 48px !important;
            min-width: 48px !important;
            min-height: 48px !important;
            max-width: 48px !important;
            max-height: 48px !important;
            border-radius: 50% !important;
            border: none !important;
            background: transparent !important;
            font-size: 60px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            margin: 0 auto !important;
            box-shadow: none !important;
            transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            line-height: 1 !important;
        }
        .stSidebar .stButton button:hover {
            transform: scale(1.06) !important;
            background: rgba(236, 64, 122, 0.06) !important;
            box-shadow: 0 0 12px rgba(236, 64, 122, 0.08) !important;
        }
        .sidebar-caption {
            text-align: center;
            color: #AD1457;
            font-weight: bold;
            font-size: 13px;
            padding-top: 3px;
            margin-bottom: 8px;
        }

        /* ----- PROFIL TIM DI SIDEBAR (KOTAK PER ANGGOTA) ----- */
        .sidebar-profile {
            margin-top: 8px;
            padding: 5px 5px;
        }
        .sidebar-profile h4 {
            color: #AD1457;
            text-align: center;
            margin-bottom: 10px;
            font-size: 1rem;
        }
        .sidebar-profile .profile-item {
            display: flex;
            align-items: center;
            margin-bottom: 14px !important;
            padding: 12px 14px;
            border-radius: 14px;
            background: #ffffff !important;
            border: 2px solid #EC407A !important;
            box-shadow: 0 4px 12px rgba(173,20,87,0.15);
            transition: 0.2s;
        }
        .sidebar-profile .profile-item:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 18px rgba(173,20,87,0.25);
        }
        .sidebar-profile .profile-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #EC407A, #D81B60);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 18px;
            margin-right: 14px;
            flex-shrink: 0;
            overflow: hidden;
            border: 2px solid white;
            box-shadow: 0 2px 8px rgba(173,20,87,0.15);
        }
        .sidebar-profile .profile-avatar img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .sidebar-profile .profile-info .name {
            font-weight: bold;
            font-size: 0.95rem;
            color: #6A1B4D;
        }
        .sidebar-profile .profile-info .detail {
            font-size: 0.75rem;
            color: #880E4F;
            margin-top: 2px;
        }
        .sidebar-university {
            text-align: center;
            padding: 8px 0;
            color: #AD1457;
            font-weight: bold;
            font-size: 0.9rem;
            border-top: 1px solid #F8BBD0;
            margin-top: 8px;
        }

        /* ----- GAYA KONTEN UTAMA ----- */
        .content-card {
            background: white;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(173,20,87,0.08);
            border: 1px solid #F8BBD0;
        }
        .content-card h2 {
            color: #AD1457;
            margin-top: 0;
        }
        .image-card {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(173,20,87,0.1);
            border: 1px solid #F8BBD0;
            margin: 0.5rem 0;
        }
        .image-card img {
            border-radius: 12px;
            border: 2px solid #F8BBD0;
            width: 100%;
        }
        .download-btn {
            background: linear-gradient(135deg, #EC407A, #D81B60);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 0.6rem 2rem;
            font-weight: bold;
            transition: 0.3s;
            box-shadow: 0 4px 15px rgba(233,30,99,0.3);
            cursor: pointer;
        }
        .download-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 8px 25px rgba(233,30,99,0.4);
        }
        .info-box {
            background: rgba(255,255,255,0.7);
            border-left: 5px solid #EC407A;
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-top: 1.5rem;
            box-shadow: 0 2px 10px rgba(233,30,99,0.08);
        }
        .info-box b {
            color: #AD1457;
        }

        /* ----- KOTAK KETERANGAN TAMBAHAN (di bawah setiap halaman) ----- */
        .footer-note {
            background: linear-gradient(135deg, #FFF9C4, #FFE082);
            border-radius: 16px;
            padding: 1.2rem 2rem;
            margin-top: 2rem;
            border: 1px solid #FFB300;
            text-align: center;
            color: #BF360C;
        }
        .footer-note p {
            margin: 0;
            font-size: 1rem;
        }

        /* ----- GAYA KHUSUS HOME (bling-bling) ----- */
        .home-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #FFF9C4, #FFE082);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 2px solid #FFB300;
            box-shadow: 0 0 30px rgba(255, 193, 7, 0.2);
            animation: glowPulse 2s ease-in-out infinite alternate;
        }
        @keyframes glowPulse {
            0% { box-shadow: 0 0 20px rgba(255, 193, 7, 0.1); }
            100% { box-shadow: 0 0 50px rgba(255, 193, 7, 0.4); }
        }
        .home-header h1 {
            font-size: 2.8rem;
            color: #E65100;
            margin: 0;
            font-weight: 900;
            text-shadow: 0 0 20px rgba(255, 193, 7, 0.3);
        }
        .bling-shower {
            font-size: 2rem;
            letter-spacing: 6px;
            color: #FFB300;
            animation: sparkle 1.5s ease-in-out infinite alternate;
        }
        @keyframes sparkle {
            0% { opacity: 0.4; transform: scale(0.95); }
            100% { opacity: 1; transform: scale(1.1); }
        }

        /* ----- GAYA KHUSUS GRAYSCALE (bunga) ----- */
        .grayscale-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid #F8BBD0;
        }
        .grayscale-header h1 {
            font-size: 2.5rem;
            color: #AD1457;
            margin: 0;
            font-weight: 800;
        }
        .flower-shower {
            font-size: 1.8rem;
            letter-spacing: 4px;
            color: #EC407A;
            animation: twinkle 2s infinite alternate;
        }
        @keyframes twinkle {
            0% { opacity: 0.6; transform: scale(1); }
            100% { opacity: 1; transform: scale(1.05); }
        }

        /* ----- GAYA KHUSUS KOMPRESI (awan) ----- */
        .kompresi-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid #90CAF9;
        }
        .kompresi-header h1 {
            font-size: 2.5rem;
            color: #0D47A1;
            margin: 0;
            font-weight: 800;
        }
        .cloud-shower {
            font-size: 1.8rem;
            letter-spacing: 4px;
            color: #42A5F5;
            animation: floatCloud 3s ease-in-out infinite alternate;
        }
        @keyframes floatCloud {
            0% { transform: translateY(0); }
            100% { transform: translateY(-8px); }
        }

        /* ----- GAYA KHUSUS DETEKSI (love) ----- */
        .deteksi-header {
            text-align: center;
            padding: 1rem 0 0.5rem 0;
            background: linear-gradient(135deg, #FCE4EC, #FFF0F5);
            border-radius: 20px;
            margin-bottom: 2rem;
            border: 1px solid #F8BBD0;
        }
        .deteksi-header h1 {
            font-size: 2.5rem;
            color: #AD1457;
            margin: 0;
            font-weight: 800;
        }
        .love-shower {
            font-size: 1.8rem;
            letter-spacing: 4px;
            color: #EC407A;
            animation: pulseLove 1.5s ease-in-out infinite alternate;
        }
        @keyframes pulseLove {
            0% { transform: scale(1); opacity: 0.7; }
            100% { transform: scale(1.1); opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# ======================== SESSION STATE ========================
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "home_visited" not in st.session_state:
    st.session_state.home_visited = False
if "grayscale_visited" not in st.session_state:
    st.session_state.grayscale_visited = False
if "kompresi_visited" not in st.session_state:
    st.session_state.kompresi_visited = False
if "deteksi_visited" not in st.session_state:
    st.session_state.deteksi_visited = False

# ======================== FUNGSI BANTU UNTUK FOTO (untuk sidebar) ========================
def get_image_base64(path_or_url):
    try:
        parsed = urlparse(path_or_url)
        if parsed.scheme in ('http', 'https'):
            response = requests.get(path_or_url, timeout=5)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode()
        else:
            if os.path.exists(path_or_url):
                with open(path_or_url, "rb") as f:
                    return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

# ======================== SIDEBAR NAVIGASI & PROFIL ========================
# --- HEADER SIDEBAR ---
st.sidebar.markdown("""
<div class="sidebar-header">
    <span class="logo">🌸</span>
    <span class="title">ANGEL</span>
    <div class="subtitle">✨ Edit & Kreasikan Gambarmu ✨</div>
</div>
""", unsafe_allow_html=True)

# --- PESAN DI BAWAH HEADER ---
st.sidebar.markdown("""
<div style="text-align: center; font-size: 14px; color: #880E4F; padding: 0 5px 8px 5px; font-style: italic;">
    Lupakan dia yang membuatmu terluka,<br>semoga web ini bisa membuatmu bahagia. <br> <br> Silahkan pilih menu yang diinginkan !
</div>
""", unsafe_allow_html=True)

# --- MENU NAVIGASI ---
menus = [
    ("🏠", "🏠 Home", "Home"),
    ("🌫️", "🌫️ Grayscale", "Grayscale"),
    ("🗜️", "🗜️ Kompresi", "Kompresi"),
    ("🔍", "🔍 Deteksi", "Deteksi")
]

cols = st.sidebar.columns(4)
for col, (emoji, page_name, label) in zip(cols, menus):
    with col:
        is_active = (st.session_state.page == page_name)
        if is_active:
            st.markdown(f"""
                <style>
                    .stSidebar .stButton button[data-testid="baseButton-secondary"]:has(> div:contains("{emoji}")) {{
                        background: #F8BBD0 !important;
                        transform: translateY(2px) scale(1.03) !important;
                        box-shadow: 0 4px 14px rgba(236,64,122,0.25) !important;
                        border: none !important;
                    }}
                </style>
            """, unsafe_allow_html=True)
        if st.button(emoji, key=f"nav_{emoji}", use_container_width=True):
            st.session_state.page = page_name
            # Reset efek visited agar balon muncul lagi
            if page_name == "🏠 Home":
                st.session_state.home_visited = False
            elif page_name == "🌫️ Grayscale":
                st.session_state.grayscale_visited = False
            elif page_name == "🗜️ Kompresi":
                st.session_state.kompresi_visited = False
            elif page_name == "🔍 Deteksi":
                st.session_state.deteksi_visited = False
            st.rerun()

# --- CAPTION DI BAWAH TOMBOL ---
st.sidebar.markdown("---")
if st.session_state.page == "🏠 Home":
    st.sidebar.markdown('<p class="sidebar-caption">🏠 Home</p>', unsafe_allow_html=True)
elif st.session_state.page == "🌫️ Grayscale":
    st.sidebar.markdown('<p class="sidebar-caption">🌫️ Grayscale</p>', unsafe_allow_html=True)
elif st.session_state.page == "🗜️ Kompresi":
    st.sidebar.markdown('<p class="sidebar-caption">🗜️ Kompresi</p>', unsafe_allow_html=True)
elif st.session_state.page == "🔍 Deteksi":
    st.sidebar.markdown('<p class="sidebar-caption">🔍 Deteksi Kemiripan</p>', unsafe_allow_html=True)

# ======================== PROFIL TIM DI SIDEBAR (dengan kotak tegas) ========================
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-profile">', unsafe_allow_html=True)
st.sidebar.markdown("### 👥 Pengembangan Aplikasi")
st.sidebar.markdown("**Teknik Informatika**")

# DATA ANGGOTA
anggota = [
    {
        "inisial": "GDA",
        "nama": "Gea Destadia Al-Zahra",
        "ig": "@gea_destadia_10",
        "telp": "0831-5068-7481",
        "foto": "assets/gea.jpg"
    },
    {
        "inisial": "LA",
        "nama": "Luna Amilia",
        "ig": "@luunaaamiiii",
        "telp": "0895-3780-96802",
        "foto": "assets/luna.jpg"
    },
    {
        "inisial": "NA",
        "nama": "Nadia Azizah",
        "ig": "@ndyyzh",
        "telp": "0858-4631-3309",
        "foto": "assets/nadia.jpg"
    },
    {
        "inisial": "DAAD",
        "nama": "Dalilah Arifah Ariandi DJR",
        "ig": "@adellianav",
        "telp": "0813-1211-6787",
        "foto": "assets/dalilah.jpg"
    },
]

for member in anggota:
    foto_b64 = get_image_base64(member.get("foto", ""))
    if foto_b64:
        avatar_html = f'<img src="data:image/jpeg;base64,{foto_b64}" />'
    else:
        avatar_html = member["inisial"]
    
    st.sidebar.markdown(f"""
    <div class="profile-item">
        <div class="profile-avatar">{avatar_html}</div>
        <div class="profile-info">
            <div class="name">• {member['nama']} •</div>
            <div class="detail">📸 {member['ig']}</div>
            <div class="detail">📞 {member['telp']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-university">🎓 Universitas Negeri Semarang</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER SIDEBAR ---
st.sidebar.markdown("""
<div class="sidebar-footer">
    🌸 Made with Love by Team ANGEL 🌸
</div>
""", unsafe_allow_html=True)

# ======================== ROUTING HALAMAN ========================
page = st.session_state.page

if page == "🏠 Home":
    home.tampilkan()
elif page == "🌫️ Grayscale":
    grayscale.tampilkan()
elif page == "🗜️ Kompresi":
    kompresi.tampilkan()
elif page == "🔍 Deteksi":
    deteksi.tampilkan()
