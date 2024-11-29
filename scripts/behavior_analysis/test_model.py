import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import cv2
from preprocess import load_and_match_data  # 전처리 함수 불러오기

# 저장된 모델 경로
model_path = 'models/dog_behavior_heading_model.keras'

# 모델 로드
model = load_model(model_path)

# 데이터 전처리 함수
def prepare_data(data_pairs, img_size=(224, 224)):
    X = []
    y = []

    for frame_path, label_data in data_pairs:
        # 이미지를 ResNet50 입력 크기에 맞게 로드
        img = load_img(frame_path, target_size=img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # 이미지 정규화 (0~1 사이 값)
        
        # JSON의 키포인트 데이터 전처리
        keypoints = []
        for _, point in label_data["annotations"][0]["keypoints"].items():
            keypoints.append([point["x"], point["y"]])
        keypoints = np.array(keypoints).flatten()

        X.append(img_array)
        y.append(keypoints)

    X = np.array(X)
    y = np.array(y)
    return X, y

# 테스트 데이터 로드 및 전처리
labeling_dir = 'dog_behavior_analysis/validation/heading/labeling_heading'
frame_dir = 'dog_behavior_analysis/validation/heading/frame_heading'
output_csv = 'annotations_heading.csv'

# load_and_match_data 호출
data_pairs = load_and_match_data(labeling_dir, frame_dir, output_csv)

# 데이터 로드 확인
if not data_pairs:
    print("data_pairs가 비어 있습니다. 경로와 데이터를 확인하세요.")
    exit()

# 데이터 전처리
X_test, y_test = prepare_data(data_pairs)


# 예측
predictions = model.predict(X_test)

# 성능 평가 및 시각화 (예시로 첫 5개 프레임 시각화)
mse_matric = tf.keras.metrics.MeanSquaredError()
mae_matric = tf.keras.metrics.MeanAbsoluteError()

mse_matric.update_state(y_test, predictions)
mae_matric.update_state(y_test, predictions)

mse = mse_matric.result().numpy()
mae = mae_matric.result().numpy()

print("Test MSE:", mse)
print("Test MAE:", mae)

# 예측 결과 시각화 (Actual과 Predicted를 동일 이미지에 시각화)
for i in range(5):
    plt.figure(figsize=(8, 8))
    
    # 강아지 이미지 시각화
    img = load_img(data_pairs[i][0])  # 원본 크기의 이미지를 로드
    img_array = img_to_array(img).astype("uint8")
    plt.imshow(img_array)
    plt.title(f"Image {i+1}")
    
    # 실제 키포인트 시각화 (빨간색)
    keypoints_actual = y_test[i].reshape(-1, 2)
    plt.scatter(keypoints_actual[:, 0], keypoints_actual[:, 1], color='red', s=60, label='Actual')
    
    # 예측된 키포인트 시각화 (파란색)
    keypoints_predicted = predictions[i].reshape(-1, 2)
    plt.scatter(keypoints_predicted[:, 0], keypoints_predicted[:, 1], color='blue', s=60, label='Predicted')
    
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 비디오 처리 함수
def process_video(video_path, model, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frame_results = []

    # 총 프레임 수 확인
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"총 프레임 수 : {total_frames}")

    frame_count = 0  # 프레임 카운터

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임을 모델 입력 형식으로 전처리
        frame_resized = cv2.resize(frame, img_size)
        frame_array = img_to_array(frame_resized)
        frame_array = frame_array / 255.0  # 이미지 정규화 (0~1 사이 값)
        frame_array = np.expand_dims(frame_array, axis=0)  # 배치 차원 추가

        # 예측 수행
        prediction = model.predict(frame_array)
        frame_results.append(prediction)

        # 첫 번째 프레임에 대해서만 시각화
        if frame_count == 0:
            plt.figure(figsize=(8, 8))
            img = frame.astype("uint8")
            plt.imshow(img)
            predicted_keypoints = prediction[0].reshape(-1, 2)
            plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], color='blue', s=60)
            plt.axis('off')
            
            # 행동 해석 및 출력
            confidence_text = interpret_pose_with_confidence(predicted_keypoints)
            print(confidence_text)  # 행동에 대한 확률 텍스트 출력
            
            # 시각화할 때 텍스트 추가
            plt.title(confidence_text)
            plt.show()
        else:
            # 나머지 프레임에서 행동 가능성 텍스트 출력
            predicted_keypoints = prediction[0].reshape(-1, 2)
            confidence_text = interpret_pose_with_confidence(predicted_keypoints)
            print(f"Frame {frame_count}: {confidence_text}")

        frame_count += 1

    cap.release()

# 행동 해석 함수
def interpret_pose_with_confidence(predicted_keypoints):
    # 다리와 몸통 키포인트의 y 좌표 차이를 계산
    leg_y = predicted_keypoints[10:12, 1].mean()  # 다리 키포인트의 평균 y 좌표
    body_y = predicted_keypoints[5:7, 1].mean()   # 몸통 키포인트의 평균 y 좌표
    
    # 거리 차이 계산
    y_diff = leg_y - body_y

    # 거리 차이를 기반으로 확률 계산
    if y_diff > 10:
        # 앉아있는 경우: y_diff가 클수록 앉아있을 확률이 높음
        confidence = min(100, y_diff * 5)
        return f"강아지가 {confidence:.1f}% 가능성으로 앉아있습니다."
    else:
        # 서있는 경우: y_diff가 작을수록 서 있을 확률이 높음
        confidence = min(100, abs(y_diff) * 5)
        return f"강아지가 {confidence:.1f}% 가능성으로 서 있습니다."

# 비디오 경로 설정
video_path = r'C:\Users\admin\IdeaProjects\test\VSCode\video_test\8.mp4'

# 비디오 처리
process_video(video_path, model)
