import streamlit as st
import pickle
import gzip
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------------
# 1. Cấu hình Trang & Giao diện
# ------------------------------
st.set_page_config(
    page_title="Movie Magic Recommender",
    page_icon="🍿",
    layout="wide"
)

# ------------------------------
# 2. Khởi tạo Session State
# ------------------------------
for key in ["history", "mode", "selected_movie", "random_movie"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key == "history" else None

# ------------------------------
# 3. TMDB API & Load Data (SỬA LỖI TẠI ĐÂY)
# ------------------------------
# Lấy API Key từ Secrets (Cách bảo mật nhất)
try:
    TMDB_API_KEY = st.secrets["tmdb"]["api_key"]
except:
    # Nếu chưa cài Secrets, dùng Key tạm thời của bạn
    TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

@st.cache_data # Dùng cache để web load nhanh hơn
def load_data():
    # Sử dụng gzip để đọc file nén giúp vượt giới hạn 100MB của GitHub
    # Đảm bảo bạn đã nén similarity.pkl thành similarity.pkl.gz trong Colab
    try:
        with open('movie_list.pkl', 'rb') as f:
            movies = pickle.load(f)
        
        # Nếu có file nén .gz thì dùng gzip, nếu không dùng pickle thường
        if os.path.exists('similarity.pkl.gz'):
            with gzip.open('similarity.pkl.gz', 'rb') as f:
                similarity = pickle.load(f)
        else:
            with open('similarity.pkl', 'rb') as f:
                similarity = pickle.load(f)
        return movies, similarity
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu: {e}")
        st.stop()

movies, similarity = load_data()

# ------------------------------
# 4. Các hàm hỗ trợ (Giữ nguyên logic của bạn)
# ------------------------------
def requests_retry_session(retries=5, backoff_factor=1):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[500, 502, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        data = requests_retry_session().get(url).json()
        return f"https://image.tmdb.org/t/p/w500{data.get('poster_path')}" if data.get('poster_path') else None
    except: return None

# ... (Các hàm fetch_trailer, get_movie_details, recommend giữ nguyên như code của bạn)

# ------------------------------
# 5. Giao diện người dùng (UI)
# ------------------------------
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Let’s Find Your Next Movie! 🎬</h1>", unsafe_allow_html=True)

# Hiển thị Trending
st.subheader("🔥 Now Trending")
trending = requests_retry_session().get(f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}").json().get("results", [])[:5]
t_cols = st.columns(5)
for i, m in enumerate(trending):
    with t_cols[i]:
        st.image(f"https://image.tmdb.org/t/p/w500{m['poster_path']}", use_container_width=True)
        st.caption(m['title'])

st.divider()

# Tìm kiếm & Surprise
c1, c2, c3 = st.columns([3, 1, 2])
with c1:
    selected_movie = st.selectbox("Search...", movies["title"].values)
    if st.button("Get Recommendations"):
        st.session_state.mode = "search"
        st.session_state.selected_movie = selected_movie
        st.rerun() # Thay cho experimental_rerun()

# ------------------------------
# 6. Sidebar & Footer
# ------------------------------
with st.sidebar:
    st.header("🕒 History")
    # Hiển thị lịch sử xem phim tại đây
