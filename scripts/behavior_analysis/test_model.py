import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter

# 행동 클래스 정의
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

# 이미지 및 키포인트 데이터 로드 및 전처리 함수
def load_and_preprocess_data(csv_path, img_size=(224, 224)):
    data = pd.read_csv(csv_path)
    X, y = [], []
    keypoints = []

    for _, row in data.iterrows():
        try:
            # 이미지 로드 (cropped_path 사용)
            img = cv2.imread(row['cropped_path'])
            img = cv2.resize(img, img_size)
            img = img / 255.0  # 정규화
            X.append(img)

            # 레이블
            y.append(row['label'])

            # 키포인트
            keypoints.append(row[['x1', 'y1', 'x3', 'y2', 'x5', 'y3', 'x7', 'y4', 'x9', 'y5', 'x11', 'y6', 'x13', 'y7', 'x15', 'y8', 'x17', 'y9', 'x19', 'y10', 'x21', 'y11', 'x23', 'y12', 'x25', 'y13', 'x27', 'y14', 'x29', 'y15']].values)
        except Exception as e:
            print(f"[WARNING] 이미지 로드 실패: {row['cropped_path']} - {e}")

    return np.array(X), np.array(y), np.array(keypoints)

# ResNet 모델 생성 함수 (키포인트를 입력으로 받을 수 있도록 수정)
def create_resnet_model(output_shape, keypoint_input_shape=(30,)):
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = True
    for layer in base_model.layers[:140]:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='softmax')
    ])
    
    # 키포인트를 추가적인 입력으로 활용하고 싶다면
    # 아래 코드를 사용하여 멀티 입력 모델로 수정 가능
    # keypoint_input = tf.keras.Input(shape=keypoint_input_shape, name='keypoints')
    # combined = tf.keras.layers.concatenate([model.output, keypoint_input])
    # combined_output = Dense(output_shape, activation='softmax')(combined)
    # model = tf.keras.models.Model(inputs=[model.input, keypoint_input], outputs=combined_output)
    
    return model

# 모델 훈련
if __name__ == "__main__":
    csv_path = r'data\split_data\annotations_train.csv'
    img_width, img_height = 125, 125

    # 데이터 로드 및 전처리
    X, y, keypoints = load_and_preprocess_data(csv_path)
    print(f"[INFO] 데이터 준비 완료. X: {X.shape}, y: {y.shape}, 키포인트: {keypoints.shape}")

    # 훈련/검증 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 모델 생성
    model = create_resnet_model(output_shape=len(behavior_classes))

    # 모델 컴파일
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 콜백 설정 (조기 종료, 모델 체크포인트 등)
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

    # 모델 훈련
    history = model.fit(
        X_train, y_train,
        batch_size=16,
        epochs=30,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    # 모델 평가
    y_pred = np.argmax(model.predict(X_val), axis=1)
    print(classification_report(y_val, y_pred, target_names=behavior_classes.values()))

    # 훈련 곡선 시각화
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

    # 혼동 행렬
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=behavior_classes.values(), yticklabels=behavior_classes.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Grad-CAM 설명 (옵션)
    def grad_cam(input_model, image, class_index, layer_name):
        grad_model = tf.keras.models.Model(
            inputs=[input_model.inputs],
            outputs=[input_model.get_layer(layer_name).output, input_model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            oss = predictions[:, class_index]
        grads = tape.gradient(oss, conv_outputs)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        heatmap = np.mean(conv_outputs.numpy()[0] * pooled_grads.numpy(), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    # Grad-CAM 실행
    heatmap = grad_cam(model, X_val[0], y_pred[0], 'conv5_block3_out')
    heatmap = cv2.resize(heatmap, (224, 224))
    plt.imshow(X_val[0])
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.show()
