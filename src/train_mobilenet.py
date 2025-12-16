import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ========= CẤU HÌNH CƠ BẢN =========
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "Fruits-360")
TRAIN_DIR = os.path.join(BASE_DIR, "Training")
VALID_DIR = os.path.join(BASE_DIR, "Test") # Fruits-360 có thư mục Test thay vì Validation
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_HEIGHT = 100   # ảnh Fruits-360 là 100x100
IMG_WIDTH = 100
BATCH_SIZE = 32
EPOCHS = 25   # có thể tăng lên 30–40 nếu máy chịu được

print("Train dir:", TRAIN_DIR)
print("Valid dir:", VALID_DIR)

# ========= DATA GENERATOR + AUGMENTATION =========
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # chuẩn hoá theo MobileNetV2
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

valid_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

valid_generator = valid_datagen.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

num_classes = train_generator.num_classes
print("Số lớp:", num_classes)

# ========= XÂY DỰNG MODEL MOBILENETV2 =========
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)

# Đóng băng backbone để chỉ train phần head phía trên
base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ========= CALLBACKS =========
checkpoint_path = os.path.join(MODEL_DIR, "fruit_mobilenet_best.h5")

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ========= TRAIN =========
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=valid_generator,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

# Lưu model cuối cùng (dù sao best model vẫn nằm ở fruit_mobilenet_best.h5)
final_path = os.path.join(MODEL_DIR, "fruit_mobilenet_final.h5")
model.save(final_path)
print("Đã lưu model cuối tại:", final_path)
print("Model tốt nhất lưu tại:", checkpoint_path)
