import os
import numpy as np
import streamlit as st
import pandas as pd # Thêm thư viện này để vẽ biểu đồ
from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==== CẤU HÌNH ====
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "Fruits-360")
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fruit_mobilenet_best.h5")

IMG_HEIGHT = 100
IMG_WIDTH = 100

# ================== CÁC HÀM XỬ LÝ (BACKEND) ==================

@st.cache_resource
def load_trained_model():
    """Load model đã train (Cache để không load lại mỗi lần f5)."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Không tìm thấy model tại: {MODEL_PATH}")
        return None

@st.cache_data
def get_class_labels():
    """
    TỐI ƯU: Đọc tên thư mục để lấy nhãn thay vì dùng ImageDataGenerator (quét ảnh rất lâu).
    flow_from_directory sắp xếp theo alpha, nên ta dùng sorted(os.listdir) là khớp.
    """
    if not os.path.exists(TRAIN_DIR):
        st.error(f"Không tìm thấy thư mục dữ liệu: {TRAIN_DIR}")
        return {}
    
    # Lấy tên các thư mục con và sắp xếp A-Z (trùng khớp với cách training)
    class_names = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    return idx_to_class

def preprocess_image(image):
    """
    Chuẩn bị ảnh cho model: Resize -> Array -> Preprocess (MobileNetV2 chuẩn)
    """
    # 1. Resize ảnh về đúng kích thước model yêu cầu
    img = image.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    
    # 2. Chuyển sang mảng numpy
    img_array = np.array(img)
    
    # 3. Mở rộng chiều (1, 100, 100, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. QUAN TRỌNG: Dùng hàm chuẩn hóa của MobileNetV2 (thay vì / 255.0)
    # Vì lúc train bạn dùng preprocess_input, lúc predict cũng phải dùng nó.
    img_array = preprocess_input(img_array)
    
    return img_array

# ================== GIAO DIỆN (FRONTEND) ==================

st.set_page_config(
    page_title="Phân loại trái cây AI", 
    page_icon="🍎", 
    layout="wide" # Dùng layout rộng để chia cột đẹp hơn
)

# --- Sidebar ---
with st.sidebar:
    st.title("🔧 Control Panel")
    st.info("Đồ án Phân loại trái cây\nMobileNetV2 Model")
    
    # Load data
    model = load_trained_model()
    idx_to_class = get_class_labels()
    
    if idx_to_class:
        st.success(f"Đã load {len(idx_to_class)} loại trái cây.")
        with st.expander("Xem danh sách class"):
            st.text("\n".join(idx_to_class.values()))

# --- Main Content ---
st.title("🍎 Nhận diện trái cây thông minh")
st.markdown("---")

# Layout chia 2 cột: Upload bên trái, Kết quả bên phải
col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Chọn hình ảnh")
    uploaded_file = st.file_uploader("Tải ảnh lên (.jpg, .png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh bạn chọn", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích")
    
    if uploaded_file is not None and model is not None:
        if st.button("🔍 Bắt đầu nhận diện", type="primary", use_container_width=True):
            with st.spinner("AI đang suy nghĩ..."):
                # Xử lý và dự đoán
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                
                # Lấy kết quả cao nhất
                top_idx = np.argmax(predictions[0])
                top_class = idx_to_class[top_idx]
                confidence = predictions[0][top_idx]
                
                # --- Hiển thị kết quả chính ---
                st.success(f"Đây là: **{top_class}**")
                st.metric(label="Độ chính xác", value=f"{confidence*100:.2f}%")
                
                # --- Hiển thị Top 5 dự đoán (Biểu đồ) ---
                st.markdown("##### Top 5 khả năng cao nhất:")
                
                # Lấy top 5 index
                top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                top_5_values = predictions[0][top_5_indices]
                top_5_labels = [idx_to_class[i] for i in top_5_indices]
                
                # Tạo dataframe cho biểu đồ
                chart_data = pd.DataFrame({
                    "Loại quả": top_5_labels,
                    "Tỉ lệ": top_5_values
                }).set_index("Loại quả")
                
                st.bar_chart(chart_data, color="#ff4b4b")
                
    elif uploaded_file is None:
        st.info("👈 Vui lòng tải ảnh lên ở cột bên trái.")