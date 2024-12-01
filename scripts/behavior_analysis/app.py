import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
import cv2
import tempfile
from stgcn_model import UnifiedModel3DCNN  # 모델 클래스
from config.config import Config
from video_behavior_analysis import calculate_behavior_duration, get_primary_behavior


# Config 인스턴스 생성
config = Config()

# Flask 애플리케이션 초기화
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploaded_videos/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 모델 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UnifiedModel3DCNN(
    in_channels=2,  # x, y 좌표
    num_joints=15,  # 관절 개수
    num_classes=12,  # 행동 클래스 개수
    num_frames=30    # 프레임 수
).to(device)

# 모델 가중치 로드
model.load_state_dict(torch.load("models/stgcn_behavior.pth", map_location=device))
model.eval()  # 평가 모드 설정

# 비디오 처리 함수
def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # 프레임 크기 조정
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise ValueError("비디오에서 프레임을 추출할 수 없습니다.")
    return np.array(frames)

# 예측 수행 함수
def predict_behavior(frames):
    skeletons = np.random.rand(len(frames), 15, 2)  # 임시 스켈레톤 데이터 생성
    skeletons = torch.tensor(skeletons, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    adjacency_matrix = torch.eye(15)
    with torch.no_grad():
        predictions = model(skeletons.to(device), adjacency_matrix.to(device))
    predicted_label = torch.argmax(predictions, dim=1).item()
    return predicted_label

# Flask 엔드포인트 정의
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400


     # 파일 저장
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"File saving error: {str(e)}"}), 500

    try:
        # 비디오 처리 및 분석
        frames = process_video(file_path)
        predicted_label = predict_behavior(frames)

        return jsonify({
            "message": "분석 성공",
            "predicted_label": predicted_label,
            "result": f"예측된 행동 클래스는 {predicted_label}입니다.",
            "video_path": file_path
        }), 200
    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)