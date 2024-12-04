import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter



# Config 클래스에서 행동 클래스 로드
behavior_classes = {
    0: '몸 낮추기',
    1: '몸 긁기',
    2: '몸 흔들기',
    3: '앞발 들기',
    4: '한쪽 발 들기',
    5: '고개 돌리기',
    6: '누워 있기',
    7: '마운팅',
    8: '앉아 있기',
    9: '꼬리 흔들기',
    10: '돌아보기',
    11: '걷거나 뛰기'
}

# 1. 데이터 샘플링 함수
def sample_behavior_data(csv_path, behavior_classes):
    """
    각 행동(class)별로 최대 n_samples_per_class만큼 샘플링하여 데이터를 반환합니다.
    """
    data = pd.read_csv(csv_path)
    sampled_data = []

    for class_id in behavior_classes.keys():
        class_data = data[data['label'] == class_id]
        if len(class_data) > n_samples_per_class:
            class_data = class_data.sample(n=n_samples_per_class, random_state=42)
        sampled_data.append(class_data)

    return pd.concat(sampled_data, ignore_index=True)

# 2. 이미지 로드 및 전처리 함수
def load_and_preprocess_image(filepath, img_size=(128, 128)):
    """이미지를 로드하고 전처리합니다."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"이미지 파일이 없습니다: {filepath}")
    img = cv2.imread(filepath)
    img = cv2.resize(img, img_size)  # ResNet50 입력 크기 맞추기
    img = img / 255.0  # 정규화
    return img

# 3. ResNet 기반 모델 생성
def create_resnet_model(output_shape):
    """ResNet50 기반 모델 생성"""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = True
    for layer in base_model.layers[:140]:  # 일부 레이어 동결
        layer.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')  # 다중 클래스 분류
    ])
    return model

# 4. 훈련 실행
if __name__ == "__main__":
    # CSV 경로 설정
    csv_path = 'data/csv_file/train_numeric_updated.csv'
    img_width, img_height = 128, 128
    n_samples_per_class = 230  # 클래스당 최대 샘플 수

    # 데이터 로드 및 샘플링
    try:
        data = sample_behavior_data(csv_path, behavior_classes)
        print(f"[INFO] 데이터 샘플링 완료. 총 샘플 수: {len(data)}")
    except Exception as e:
        print(f"[ERROR] 데이터 로드 중 에러 발생: {e}")
        exit()

    # 이미지와 라벨 데이터 생성
    X, y = [], []
    for _, row in data.iterrows():
        try:
            img = load_and_preprocess_image(row['frame_path'], (img_width, img_height))
            X.append(img)
            y.append(row['label'])
        except Exception as e:
            print(f"[WARNING] 이미지 로드 실패: {row['frame_path']} - {e}")

    # Numpy 배열로 변환
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"[INFO] 데이터 준비 완료. X: {X.shape}, y: {y.shape}")

    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 모델 생성 및 컴파일
    model = create_resnet_model(output_shape=len(behavior_classes))
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 콜백 설정
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('dog_behavior_model.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

        # 데이터 증강 설정
    datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

    # 클래스 가중치 계산
    class_weights = {i: len(y) / (len(behavior_classes) * Counter(y)[i]) for i in range(len(behavior_classes))}


    # 모델 훈련
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # 평가 결과 출력
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print(classification_report(y_val, y_pred, target_names=behavior_classes.values()))

    # 학습 곡선 시각화
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.show()


    # 훈련 및 검증 정확도/손실 시각화
    plt.figure(figsize=(12, 6))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


    # 혼동 행렬 계산
    cm = confusion_matrix(y_val, y_pred)

    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=behavior_classes.values(), yticklabels=behavior_classes.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()



    from sklearn.metrics import classification_report

    # Classification Report 계산
    report = classification_report(y_val, y_pred, target_names=behavior_classes.values(), output_dict=True)

    # Precision, Recall, F1-score 시각화
    df_report = pd.DataFrame(report).transpose()

    # 막대 그래프 생성
    df_report[:-3].plot(kind='bar', figsize=(12, 6))
    plt.title('Class-wise Performance Metrics (Precision, Recall, F1-score)')
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


    def grad_cam(input_model, image, class_index, layer_name):
    
        grad_model = tf.keras.models.Model(
            inputs=[input_model.inputs],
            outputs=[input_model.get_layer(layer_name).output, input_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            oss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        heatmap = np.mean(conv_outputs.numpy()[0] * pooled_grads.numpy(), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        return heatmap

# Grad-CAM 실행
image = X_val[0]  # 예시 이미지
true_label = y_val[0]
pred_label = y_pred[0]

heatmap = grad_cam(model, image, pred_label, 'conv5_block3_out')  # 마지막 Convolutional Layer
heatmap = cv2.resize(heatmap, (128, 128))

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title(f"Original Image (True: {behavior_classes[true_label]})")

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Heatmap overlay
plt.title(f"Grad-CAM (Predicted: {behavior_classes[pred_label]})")
plt.show()