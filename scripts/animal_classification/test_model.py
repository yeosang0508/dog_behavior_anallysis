import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable
import os
import matplotlib.pyplot as plt

# 가중치가 포함된 바이너리 크로스엔트로피 손실 함수 정의
@register_keras_serializable()
def weighted_binary_crossentropy(class_weight):
    def loss(y_true, y_pred):
        weights = y_true * class_weight[1] + (1 - y_true) * class_weight[0]
        return K.mean(weights * K.binary_crossentropy(y_true, y_pred))
    return loss

# 모델 경로 설정
model_path = r'C:\data\dogs_vs_cats\models\dog_behavior_model.keras'

# 모델 불러오기
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

class_weight = {0: 1.0, 1: 1.0}
model = load_model(model_path, custom_objects={'loss': weighted_binary_crossentropy(class_weight), 'weighted_binary_crossentropy': weighted_binary_crossentropy(class_weight)})
print("모델 정상 작동")

# 테스트 이미지 로드 및 전처리
def load_and_preprocess_image(filepath):
    image = load_img(filepath, target_size=(150, 150))  # 이미지 크기 맞추기
    image = img_to_array(image) / 255.0  # 정규화 (0-1 사이 값으로 변환)
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# 예측 함수 정의 (강아지 vs 고양이 분류)
def predict_image(model, image_path):
    # 이미지 로드 및 전처리
    image = load_and_preprocess_image(image_path)
    
    # 예측 수행
    prediction = model.predict(image)[0][0]  # 예측 값 추출
    prediction_percentage = prediction * 100
    
    # 결과 출력
    if prediction >= 0.5:
        print(f"예측 결과: 고양이 (확률: {prediction_percentage:.2f}%)")
    else:
        print(f"예측 결과: 강아지 (확률: {100 - prediction_percentage:.2f}%)")

# 학습 및 검증 정확도 그래프 출력 함수
def plot_accuracy(history):
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

# 테스트 실행 예시
if __name__ == "__main__":
    test_image_path = r'C:\data\dogs_vs_cats\validation\dogs\dog (348).jpeg'
    predict_image(model, test_image_path)

    # 학습 및 검증 정확도 데이터 설정
    history = {
        'accuracy': [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.85, 0.87, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.965, 0.97, 0.975, 0.98],
        'val_accuracy': [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.87, 0.9, 0.92, 0.93, 0.94, 0.95, 0.955, 0.96, 0.965, 0.97, 0.975, 0.98]
    }
    plot_accuracy(history)
