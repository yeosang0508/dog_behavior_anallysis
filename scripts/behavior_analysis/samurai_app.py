from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
from collections import Counter
from samurai import SamuraiTracker  # SAMURAI 라이브러리
from behavior_model import BehaviorClassificationModel  # 훈련된 모델 클래스

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


# 모델 클래스 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "best_model.pth"
model = BehaviorClassificationModel.load_model(model_path).to(device)

# SAMURAI 객체 추적기 초기화
tracker = SamuraiTracker()

# Flask 엔드포인트 정의
@app.route('/analyze', methods=['POST'])
def analyze_video():
    try:
        # 파일 가져오기
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "파일이 전달되지 않았습니다."}), 400

        # 비디오 처리 및 행동 분석
        behavior_results = process_video(file)
        if behavior_results is None or len(behavior_results) == 0:
            return jsonify({"error": "행동 분석 실패"}), 400

        # 행동 빈도 계산
        behavior_counts = Counter(behavior_results)
        total_frames = len(behavior_results)
        behavior_percentages = [
            {"behavior": behavior, "percentage": round((count / total_frames) * 100, 2)}
            for behavior, count in behavior_counts.items()
        ]

        # 가장 빈도 높은 행동 추출
        most_common_behavior = behavior_counts.most_common(1)[0][0]

        # 응답 데이터 생성
        response = {
            "frame_by_frame_behavior": behavior_results,
            "behavior_percentages": behavior_percentages,
            "most_common_behavior": most_common_behavior
        }
        return jsonify(response), 200

    except Exception as e:
        # 오류 처리
        return jsonify({"error": f"서버 오류: {str(e)}"}), 500

def process_video(file):
    # 비디오 파일 열기
    file_path = f"temp_video/{file.filename}"
    file.save(file_path)
    cap = cv2.VideoCapture(file_path)

    behavior_results = []

    # 첫 번째 프레임 읽기 및 강아지 감지
    success, frame = cap.read()
    if not success:
        return None

    # 첫 번째 프레임에서 강아지 감지
    dog_boxes = detect_dog_in_frame(frame)
    if not dog_boxes:
        return None

    # 첫 번째 프레임에서 감지된 강아지 바운딩 박스 (첫 번째 x, y, w, h)
    first_frame_bbox = dog_boxes[0]  # 첫 번째 강아지 객체의 바운딩 박스
    tracker.init(frame, first_frame_bbox)  # SAMURAI 추적기 초기화

    # 두 번째 프레임부터 처리 시작
    while success:
        # SAMURAI 객체 감지 실행
        detection_result = tracker.track(frame)
        
        if detection_result["detection_success"]:
            # 객체 감지된 영역 크롭
            bbox = detection_result["bbox"]
            cropped_frame = crop_frame(frame, bbox)
            keypoints = detection_result["keypoints"]
            behavior_class = classify_behavior(cropped_frame, keypoints)

            # 예측된 행동을 리스트에 추가
            behavior_results.append(behavior_classes[behavior_class])

        success, frame = cap.read()

    cap.release()
    return behavior_results

def crop_frame(frame, bbox):
    x, y, w, h = bbox
    return frame[int(y):int(y + h), int(x):int(x + w)]

def classify_behavior(cropped_frame, keypoints):
    input_data = prepare_input(cropped_frame, keypoints)
    output = model(input_data)
    return torch.argmax(output).item()

def prepare_input(cropped_frame, keypoints):
    # Keypoints와 크롭된 이미지를 합쳐서 입력 데이터를 만듦
    flattened = keypoints.flatten()
    return torch.tensor(flattened, dtype=torch.float32).unsqueeze(0).to(device)

def detect_dog_in_frame(frame):
    # COCO 객체 감지 모델을 통해 강아지 감지 (x, y, w, h 바운딩 박스)
    net = cv2.dnn.readNetFromCaffe('MobileNet/deploy.prototxt', 'MobileNet/mobilenet_iter_73000.caffemodel')
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

if __name__ == '__main__':
    app.run(debug=True)
