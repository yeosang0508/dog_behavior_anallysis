import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 모델 파일 경로
model_paths = {
    "sit": r"C:\Users\admin\IdeaProjects\test\VSCode\models\dog_behavior_sit_model.keras",
    "bodyscratch": r"C:\Users\admin\IdeaProjects\test\VSCode\models\dog_behavior_bodyscratch_model.keras",
    "bodyshake": r"C:\Users\admin\IdeaProjects\test\VSCode\models\dog_behavior_bodyshake_model.keras",
    "feetup": r"C:\Users\admin\IdeaProjects\test\VSCode\models\dog_behavior_feetup_model.keras",
}

# 모델 로드
models = {behavior: load_model(path) for behavior, path in model_paths.items()}

# 1. 영상에서 프레임 추출
def extract_frames(video_path, output_folder, frame_rate=1):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Frames extracted to {output_folder}")

# 2. 프레임 전처리
def preprocess_frame(frame_path, target_size=(224, 224)):
    image = load_img(frame_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # 정규화
    return image_array

# 3. 프레임 분석
def analyze_behavior(frames_folder, models):
    frame_files = sorted(os.listdir(frames_folder))
    results = []

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = preprocess_frame(frame_path)

        predictions = {}
        for behavior, model in models.items():
            prediction = model.predict(np.expand_dims(frame, axis=0))
            predictions[behavior] = prediction[0][0]  # 클래스 확률 값

        results.append({"frame": frame_file, "predictions": predictions})
    return results

# 4. 결과 요약
def summarize_behavior(results):
    behavior_counts = {behavior: 0 for behavior in models.keys()}
    
    for result in results:
        for behavior, score in result["predictions"].items():
            if score > 0.5:  # 임계값 설정 (0.5 예시)
                behavior_counts[behavior] += 1

    return max(behavior_counts, key=behavior_counts.get)  # 가장 자주 나온 행동 반환

# 메인 함수
if __name__ == "__main__":
    # 업로드된 영상 경로
    video_path = r'"C:\Users\juns0\Downloads\강아지 동영상\stand.mp4"'
    frames_folder = "frames"

    # 1. 프레임 추출
    extract_frames(video_path, frames_folder, frame_rate=10)

    # 2. 프레임 분석
    results = analyze_behavior(frames_folder, models)

    # 3. 결과 요약
    dominant_behavior = summarize_behavior(results)
    print(f"주요 강아지 행동: {dominant_behavior}")
