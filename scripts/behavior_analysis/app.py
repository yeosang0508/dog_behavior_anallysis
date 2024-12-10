import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
import cv2
from collections import Counter
import tempfile
from absl import logging

# Mediapipe 경고 메시지 비활성화
logging.set_verbosity(logging.ERROR)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 행동 클래스 정의
behavior_classes = {
    0: "🌟 몸 흔들기 - 흥 폭발! 나만의 댄스 무브~",
    1: "🔍 고개 돌리기 - '어디선가 소리가?' 집중 모드 ON!",
    2: "😴 누워 있기 - '나 좀 내버려 둬...' 완벽한 릴렉스~",
    3: "🚀 마운팅 - 하이퍼 에너지 풀가동!",
    4: "🪑 앉아 있기 - 댕댕의 시그니처 포즈, '멍!'"
}


# MobileNet SSD 모델 로드
net = cv2.dnn.readNetFromCaffe('MobileNet/deploy.prototxt', 'MobileNet/mobilenet_iter_73000.caffemodel')

# Keypoint 모델 정의
class KeypointModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeypointModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(self.bn1(x))
        x = self.fc2(x)
        x = nn.ReLU()(self.bn2(x))
        x = self.fc3(x)
        x = nn.ReLU()(self.bn3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# 모델 로드
model = KeypointModel(input_size=37, num_classes=5).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Flask 애플리케이션 생성
app = Flask(__name__)

# 강아지 감지 함수
def detect_dog_in_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    dog_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 12:  # 강아지 클래스
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")
                if x2 > x1 and y2 > y1:
                    dog_boxes.append((x1, y1, x2 - x1, y2 - y1))
    return dog_boxes

# 비디오 처리 함수
def process_video(file):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        file.save(temp_file.name)
        cap = cv2.VideoCapture(temp_file.name)
        if not cap.isOpened():
            raise ValueError("비디오 파일을 열 수 없습니다.")

        keypoints = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dog_boxes = detect_dog_in_frame(frame)
            if dog_boxes:
                keypoints.append(np.random.rand(37)) 
        cap.release()
    finally:
        temp_file.close()
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)

    if not keypoints:
        raise ValueError("키포인트 추출 실패")

    return keypoints

# API 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        file = request.files.get('file')
        if not file:
            raise ValueError("파일이 전달되지 않았습니다.")

        keypoints = process_video(file)
        input_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            predicted_classes = torch.argmax(output, dim=1).tolist()

        predicted_behaviors = [behavior_classes.get(cls, "알 수 없는 행동") for cls in predicted_classes]
        behavior_counts = Counter(predicted_behaviors)
        total_frames = len(predicted_behaviors)
        behavior_percentages = [{"behavior": behavior, "percentage": round((count / total_frames) * 100, 2)} for behavior, count in behavior_counts.items()]
        most_common_behavior = behavior_counts.most_common(1)[0][0]

        return jsonify({
            "frame_by_frame_behavior": predicted_behaviors,
            "behavior_percentages": behavior_percentages,
            "most_common_behavior": most_common_behavior
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
