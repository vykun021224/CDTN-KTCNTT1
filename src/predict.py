import os
import numpy as np
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

# ====== CẤU HÌNH ĐƯỜNG DẪN ======
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "Fruits-360")
TRAIN_DIR = os.path.join(BASE_DIR, "Training")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fruit_mobilenet_best.h5")

IMG_HEIGHT = 100
IMG_WIDTH = 100

# ====== LOAD MODEL ======
print("Đang load model từ:", MODEL_PATH)
model = load_model(MODEL_PATH)

# ====== LẤY DANH SÁCH NHÃN (LABEL) TỪ THƯ MỤC TRAIN ======
print("Đang đọc class indices từ Training/ ...")
datagen = ImageDataGenerator(rescale=1.0/255)

generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

class_indices = generator.class_indices          # {'Apple Braeburn': 0, 'Banana': 1, ...}
idx_to_class = {v: k for k, v in class_indices.items()}

print("Số lớp:", len(idx_to_class))

# ====== HÀM DỰ ĐOÁN 1 ẢNH ======
def predict_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

    # Đọc ảnh bằng OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    # Resize và chuẩn hóa
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)   # shape: (1, 100, 100, 3)

    # Dự đoán
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    pred_class = idx_to_class[pred_idx]
    confidence = float(preds[0][pred_idx])

    return pred_class, confidence


if __name__ == "__main__":
    # ====== ĐƯỜNG DẪN ẢNH TEST ======
    # Cách đơn giản: COPY 1 ảnh từ thư mục Test/ vào thư mục src/test_images/,
    # rồi sửa đường dẫn bên dưới cho đúng tên file.

    test_image_path = os.path.join(
        os.path.dirname(__file__),
        "test_images",
        "sample.png"      # đổi thành tên file ảnh thật của bạn
    )

    print("Dự đoán cho ảnh:", test_image_path)
    try:
        label, conf = predict_image(test_image_path)
        print(f"Kết quả dự đoán: {label} (độ tin cậy: {conf:.4f})")
    except Exception as e:
        print("Lỗi khi dự đoán:", e)
