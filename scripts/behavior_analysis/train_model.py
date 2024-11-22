import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config.config import SingleModelConfig
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd



# 이미지 로드 및 전처리 함수 (OpenCV 활용)
def load_and_preprocess_image(filepath):
    """이미지를 로드하고 전처리합니다."""
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"이미지 파일을 로드할 수 없습니다: {filepath}")
    img = cv2.resize(img, (224, 224))  # ResNet50 입력 크기 맞추기
    img = img / 255.0  # 정규화
    return img

# CSV에서 데이터를 불러오는 함수
def load_data_from_csv(output_csv):
    """CSV 파일에서 데이터를 불러옵니다."""
    data = pd.read_csv(output_csv)
    X = []
    y = []

    for _, row in data.iterrows():
        frame_path = row['frame_path']
        keypoints = row[2:].values  # x1, y1, ..., x15, y15

        # 이미지 로드
        if os.path.exists(frame_path):
            try:
                img = load_and_preprocess_image(frame_path)
                X.append(img)
                y.append(keypoints)
            except ValueError as e:
                print(f"Error: {e}")
        else:
            print(f"이미지 파일이 존재하지 않습니다: {frame_path}")
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)  # float32로 변환
    print(f"데이터 로드 완료. X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# ResNet50 기반 모델 구축
def create_resnet_model(output_shape):
    """ResNet50 기반의 모델을 생성합니다."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(output_shape, activation='linear')
    ])
    return model

if __name__ == "__main__":

    # CSV 파일 경로 설정
    output_csv = 'csv_file/annotations_heading.csv'

    # CSV에서 데이터 불러오기
    try:
        X, y = load_data_from_csv(output_csv)
    except Exception as e:
        print(f"데이터 로드 중 에러 발생: {e}")
        exit()

    # 모델 생성
    try:
        model = create_resnet_model(output_shape=y.shape[1])
        model.compile(optimizer=Adam(), loss='mse', metrics=['mae', 'mse'])
    except Exception as e:
        print(f"모델 생성 중 에러 발생: {e}")
        exit()

    # 학습 데이터와 검증 데이터 분리
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")
    except Exception as e:
        print(f"데이터 분할 중 에러 발생: {e}")
        exit()

    # 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('dog_behavior_analysis_model.keras', save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # 모델 훈련
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=16,
            epochs=30,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )
    except Exception as e:
        print(f"모델 훈련 중 에러 발생: {e}")
        exit()

    # 훈련 결과 시각화
    try:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['mse'], label='Train MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.title('Training and Validation MSE')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.show()
    except Exception as e:
        print(f"훈련 결과 시각화 중 에러 발생: {e}")

    # 모델 저장
    try:
        save_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'dog_behavior_heading_model.keras')
        model.save(model_path)
        print(f"모델이 저장되었습니다: {model_path}")
    except Exception as e:
        print(f"모델 저장 중 에러 발생: {e}")