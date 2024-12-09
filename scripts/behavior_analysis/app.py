from flask import Flask, request, jsonify
import cv2
import torch
import os
import numpy as np
from collections import Counter
import tempfile

app = Flask(__name__)

# 행동 클래스 정의
behavior_classes = {
    0: "몸 낮추기 - 반려견이 몸을 낮추는 동작으로, 방어적인 자세일 수 있습니다.",
    1: "몸 긁기 - 반려견이 몸을 긁고 있어요. 가려운 부위를 긁는 행동입니다.",
    2: "몸 흔들기 - 몸에 물이나 먼지가 있을 때 흔들어 털어내는 동작입니다.",
    3: "앞발 들기 - 앞발 두 개를 들고 주위를 관찰하거나 호기심을 나타냅니다.",
    4: "한쪽 발 들기 - 한쪽 발을 드는 행동으로, 긴장하거나 흥미를 느낄 때 보입니다.",
    5: "고개 돌리기 - 머리를 돌려 주위를 살피거나 관찰하는 행동입니다.",
    6: "누워 있기 - 반려견이 편안하게 누워 휴식하거나 쉬고 있는 모습입니다.",
    7: "마운팅 - 다른 개나 물체에 올라타는 동작으로, 흥분하거나 놀이 중일 수 있습니다.",
    8: "앉아 있기 - 반려견이 앉아있는 자세로, 안정된 상태를 나타냅니다.",
    9: "꼬리 흔들기 - 꼬리를 흔드는 행동으로, 기쁨이나 관심을 표현합니다.",
    10: "꼬리 낮추기 - 꼬리를 낮춘 상태로, 긴장하거나 복종을 나타낼 수 있습니다.",
    11: "돌아보기 - 몸을 돌려 주변을 확인하거나 환경을 살피는 동작입니다.",
    12: "걷거나 뛰기 - 활기차게 걷거나 뛰며 에너지가 넘치는 상태를 보여줍니다."
}

# MobileNet SSD 모델 로드
net = cv2.dnn.readNetFromCaffe('MobileNet/deploy.prototxt', 'MobileNet/mobilenet_iter_73000.caffemodel')

# 모델 클래스 정의
class KeypointModel(torch.nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64, output_dim=12):
        super(KeypointModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KeypointModel(input_dim=34, hidden_dim=64, output_dim=12).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("모델 로드 성공!")

# 비디오 파일에서 강아지 감지
def detect_dog_in_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    dog_boxes = []  # 감지된 강아지 바운딩 박스를 저장할 리스트
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:  # 신뢰도 임계값
            class_id = int(detections[0, 0, i, 1])
            if class_id == 12:  # 강아지 클래스
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")
                dog_boxes.append((x1, y1, x2 - x1, y2 - y1))  # (x, y, w, h)
    return dog_boxes

# 비디오에서 프레임을 추출하고 분석하는 함수
def process_video(file):
    # 파일을 임시 디렉토리에 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)
    
    cap = cv2.VideoCapture(temp_file.name)
    keypoints = []  # 각 프레임의 키포인트 정보를 저장할 리스트

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 비디오 끝

        # 강아지 감지
        dog_boxes = detect_dog_in_frame(frame)

        # 키포인트 처리 (예시로 랜덤값 사용, 실제로는 모델을 사용하여 추출)
        keypoints.append(np.random.rand(34))  # 실제로는 추출된 키포인트를 사용

        frame_count += 1

    cap.release()
    os.remove(temp_file.name)  # 임시 파일 삭제
    return keypoints

# Flask 엔드포인트 정의
@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        # 파일 가져오기
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "파일이 전달되지 않았습니다."}), 400

        # 비디오 처리 및 키포인트 추출
        keypoints = process_video(file)
        if keypoints is None or len(keypoints) == 0:
            return jsonify({"error": "키포인트 생성 실패"}), 400

        # 모델 입력 데이터 생성
        input_tensor = torch.tensor(keypoints, dtype=torch.float32).to(device)

        # 모델 추론
        with torch.no_grad():
            output = model(input_tensor)
            predicted_classes = torch.argmax(output, dim=1).tolist()

        # 예측된 행동 매핑
        predicted_behaviors = [
            behavior_classes.get(cls, "알 수 없는 행동") for cls in predicted_classes
        ]

        # 행동 빈도 계산
        total_frames = len(predicted_behaviors)
        behavior_counts = Counter(predicted_behaviors)

        # 행동 퍼센테이지 계산
        behavior_percentages = [
            {"behavior": behavior, "percentage": round((count / total_frames) * 100, 2)}
            for behavior, count in behavior_counts.items()
        ]

        # 가장 빈도 높은 행동 추출
        most_common_behavior = behavior_counts.most_common(1)[0][0]

        # 응답 데이터 생성
        response = {
            "frame_by_frame_behavior": predicted_behaviors,
            "behavior_percentages": behavior_percentages,
            "most_common_behavior": most_common_behavior,
            "keypoints": keypoints,
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
