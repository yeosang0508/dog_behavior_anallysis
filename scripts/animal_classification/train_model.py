import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# 데이터 로드 (preprocess.py에서 생성된 데이터 사용)
from preprocess import get_data
train_images, val_images, train_labels, val_labels = get_data()

print(f"훈련 이미지 개수: {len(train_images)}")
print(f"검증 이미지 개수: {len(val_images)}")

# OpenCV로 이미지를 로드하고 전처리하는 함수 정의
def load_and_preprocess_image(filepath):
    img = cv2.imread(filepath)
    img = cv2.resize(img, (150, 150))  # ResNet50 입력 크기 맞추기
    img = img / 255.0  # 정규화
    return img

# OpenCV 기반 증강 함수 정의
def augment_image(image):
    # 1. 밝기 조절 (alpha 범위는 0.8~1.2로 조정)
    brightness = np.random.uniform(0.8, 1.2)
    image = cv2.convertScaleAbs(image, alpha=brightness)

    # 2. 회전 (각도 범위 -30도~30도)
    angle = np.random.uniform(-30, 30)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 3. 대비 조정 (1.0~1.3 범위)
    contrast = np.random.uniform(1.0, 1.3)
    image = cv2.addWeighted(image, contrast, np.zeros_like(image), 0, 0)

    # 4. 노이즈 추가 (픽셀 범위 조정)
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    # 5. 이동 (폭과 높이의 ±10% 범위)]
    (h, w) = image.shape[:2]
    tx = np.random.uniform(-0.1 * w, 0.1 * w)
    ty = np.random.uniform(-0.1 * h, 0.1 * h)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # 6. 확대/축소 (0.9배~1.1배 범위)
    zoom = np.random.uniform(0.9, 1.1)
    new_w, new_h = int(w * zoom), int(h * zoom)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 확대 후 크기 조정 (224x224로 맞춤)
    if zoom < 1.0:
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        image = cv2.copyMakeBorder(image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT_101)
    else:
        image = cv2.resize(image, (224, 224))

    # 7. 수평 뒤집기 (50% 확률)
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    # 8. 정규화 (ResNet50 입력과 동일)
    image = image / 255.0  # 0~1로 정규화
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

   # 색조 변화 추가 (Hue shift)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_shift = np.random.uniform(-10, 10)
    image[:, :, 0] = np.clip(image[:, :, 0] + hue_shift, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

# Custom generator 정의 (OpenCV 증강 포함)
def custom_generator(generator):
    while True:
        batch = next(generator)
        images, labels = batch
        augmented_images = np.array([augment_image(img) for img in images])
        yield augmented_images, labels


# 폴더 내 모든 이미지 경로를 가져오는 함수
def load_dataset_from_directory(directory):
    images = []
    labels = []
    for class_label, class_name in enumerate(['dogs', 'cats']):  # 두 클래스
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):  # 경로가 유효한지 확인
            raise FileNotFoundError(f"Directory not found: {class_dir}")
        print(f"Processing class: {class_name} at {class_dir}")

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            print(f"Loading image: {img_path}")

            try:
                img = load_and_preprocess_image(img_path)
                images.append(img)
                labels.append(class_label)
            except Exception as e:
                print(f"Failed to load {img_name}: {e}")

    return np.array(images), np.array(labels)

def get_data():
    # 데이터셋의 경로 반환
    train_dir = "dog_behavior_analysis/train"
    val_dir = "dog_behavior_analysis/validation"
    return train_dir, val_dir

train_dir, val_dir = get_data()

# 훈련 및 검증 데이터 로드
x_train, y_train = load_dataset_from_directory(train_dir)
x_val, y_val = load_dataset_from_directory(val_dir)

# TensorFlow 데이터셋으로 변환
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32).shuffle(buffer_size=1000)
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

# ResNet50 모델 불러오기 (사전 학습된 가중치 사용)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 사전 학습된 가중치 고정
base_model.trainable = True
for layer in base_model.layers[:50]:
    layer.trainable = False

# 모델 구성
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # 유닛 수를 128로 줄임
    Dropout(0.5),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),   # 유닛 수를 64로 줄임
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weight))

# 모델 체크포인트 설정
save_dir = os.path.join(os.getcwd(), 'models')
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, 'dog_behavior_model.keras')

def weighted_binary_crossentropy(class_weight):
    def loss(y_true, y_pred):
        weights = y_true * class_weight[1] + (1 - y_true) * class_weight[0]
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred))
    return loss

optimizer = Adam(learning_rate=1e-5)
model.compile(
    loss=weighted_binary_crossentropy(class_weight),
    optimizer=optimizer,
    metrics=['accuracy']
)

# 조기 종료 콜백
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')


lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7
)

# 모델 훈련
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=val_dataset,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, lr_scheduler]
)

# 정확도 그래프 출력
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# 모델 저장
model.save(model_path)
print(f"모델 훈련 완료 및 저장 완료: {model_path}")