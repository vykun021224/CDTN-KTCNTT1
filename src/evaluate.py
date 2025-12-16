import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==== CẤU HÌNH ====
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "Fruits-360")
TEST_DIR = os.path.join(BASE_DIR, "Test")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fruit_mobilenet_best.h5")
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

def evaluate():
    # 1. Load dữ liệu Test
    print("Đang tải dữ liệu Test...")
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # Quan trọng: Không xáo trộn để đối chiếu nhãn đúng
    )

    # 2. Load Model
    print("Đang load model...")
    model = load_model(MODEL_PATH)

    # 3. Dự đoán
    print("Đang chạy dự đoán (có thể mất vài phút)...")
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes # Nhãn thực tế

    # 4. In báo cáo chi tiết (Precision, Recall, F1-Score)
    class_labels = list(test_generator.class_indices.keys())
    print("\n=== BÁO CÁO PHÂN LOẠI ===")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    # 5. Vẽ Ma trận nhầm lẫn (Confusion Matrix) - Lấy 10 class đầu tiên cho dễ nhìn
    # (Vì 131 class vẽ ra sẽ bị rối, nên ta chỉ vẽ mẫu đại diện)
    print("Đang vẽ biểu đồ...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Lấy 10 lớp đầu tiên để vẽ demo
    cm_subset = cm[:10, :10] 
    labels_subset = class_labels[:10]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels_subset, yticklabels=labels_subset)
    plt.title('Confusion Matrix (10 loại trái cây đầu tiên)')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.tight_layout()
    plt.show() # Chụp màn hình cái này đưa vào báo cáo

if __name__ == "__main__":
    evaluate()