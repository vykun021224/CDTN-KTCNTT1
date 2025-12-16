# BÁO CÁO ĐỒ ÁN: PHÂN LOẠI TRÁI CÂY (FRUIT CLASSIFICATION)

**Sinh viên thực hiện:** Lê Tấn Vỹ
**MSSV:** 2200008084
**Lớp/Học phần:** [22DTH2A]


## 1. Giới thiệu (Introduction)
Dự án này xây dựng một mô hình Deep Learning để nhận diện và phân loại hình ảnh các loại trái cây (sử dụng bộ dữ liệu Fruits-360).
Dự án bao gồm trọn bộ source code từ khâu tiền xử lý dữ liệu, huấn luyện mô hình (training) đến kiểm thử và dự đoán (testing).

## 2. Cấu trúc thư mục (Project Structure)
Cấu trúc cây thư mục của dự án như sau:

FRUIT-CLASSIFICATION/
├── dataset/             # Thư mục chứa dữ liệu ảnh (Tải từ Google Drive)
├── models/              # Chứa các file model đã train (.h5, .pt)
├── src/                 # Source code chính
│   ├── train_mobilenet.py # Code huấn luyện mô hình
│   ├── predict.py       # Code dự đoán/kiểm thử
│   └── evaluate.py      # Vẽ biểu đồ đánh giá mô hình
├── venv/                # Môi trường ảo (Đã được ignore khỏi git)
├── requirements.txt     # Danh sách các thư viện cần thiết
└── README.md            # File báo cáo hướng dẫn này

## 3. Cài đặt môi trường (Installation)
Để giảng viên/người dùng có thể chạy được dự án, vui lòng cài đặt các thư viện cần thiết theo các bước sau:

Bước 1: Clone project hoặc tải source code về máy.

Bước 2: Cài đặt các thư viện phụ thuộc. Mở terminal (CMD/PowerShell) tại thư mục gốc của dự án và chạy lệnh:

Bash

pip install -r requirements.txt
(Lưu ý: Yêu cầu Python 3.8 trở lên)

## 4. Dữ liệu (Dataset)
Do bộ dữ liệu hình ảnh có dung lượng lớn (>100MB) nên không được upload trực tiếp lên GitHub. Vui lòng tải bộ dữ liệu từ Google Drive theo đường dẫn dưới đây:

👉 LINK TẢI DATASET: [https://drive.google.com/drive/u/0/folders/1Y_QD-bGbrKTBAzI9PGJZLgtAiQk8kSSC]

Hướng dẫn setup data:

Tải file/thư mục từ link trên về máy.

Giải nén (nếu có).

Đảm bảo tên thư mục là dataset và đặt nó vào thư mục gốc của dự án (ngang hàng với thư mục src và file requirements.txt).