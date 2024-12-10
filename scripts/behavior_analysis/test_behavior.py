import os
import cv2
import numpy as np
import torch
from collections import Counter
from matplotlib import font_manager, rc
from mmdet.apis import init_detector, inference_detector  # MMSDetection 관련 라이브러리

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 행동 클래스 정의
behavior_classes = {
    0: "몸 흔들기",
    1: "고개 돌리기",
    2: "누워 있기",
    3: "마운팅",
    4: "앉아 있기",
}

# MMSDetection 모델 초기화
mms_config = "configs/mmpose/dog_pose_estimation.py"  # MMSDetection 모델 설정 파일
mms_checkpoint = "checkpoints/dog_pose.pth"  # MMSDetection 가중치 파일
pose_model = init_detector(mms_config, mms_checkpoint, device=device)

# MobileNet SSD 모델 로드
net = cv2.dnn.readNetFromCaffe(
    'MobileNet/deploy.prototxt',
    'MobileNet/mobilenet_iter_73000.caffemodel'
)

# Keypoint 모델 정의
class KeypointModel(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(KeypointModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc4 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.ReLU()(self.bn1(x))
        x = self.fc2(x)
        x = torch.nn.ReLU()(self.bn2(x))
        x = self.fc3(x)
        x = torch.nn.ReLU()(self.bn3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# 모델 로드
model = KeypointModel(input_size=45, num_classes=5).to(device)
model.load_state_dict(torch.load("C:\\Users\\admin\\IdeaProjects\\test\\VSCode\\best_model.pth", map_location=device))
model.eval()
print("모델 로드 성공!")

# MMSDetection을 이용한 키포인트 추출
def extract_keypoints_with_mms(frame, box):
    x1, y1, w, h = box
    cropped_frame = frame[y1:y1+h, x1:x1+w]
    result = inference_detector(pose_model, cropped_frame)

    # 결과에서 키포인트 추출
    keypoints = result[0]['keypoints']  # MMSDetection 결과의 키포인트 정보
    return keypoints  # [[x1, y1, score1], [x2, y2, score2], ...]

# MobileNet SSD를 이용한 강아지 감지
def detect_dog_in_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    dog_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            if class_id == 12:  # 강아지 클래스
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x1, y1, x2, y2) = box.astype("int")
                dog_boxes.append((x1, y1, x2 - x1, y2 - y1))
    return dog_boxes

# 비디오 처리 및 키포인트 추출
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    keypoints = []
    frame_data = []
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        dog_boxes = detect_dog_in_frame(frame)
        if dog_boxes:
            for box in dog_boxes:
                # MMSDetection으로 키포인트 추출
                keypoints_frame = extract_keypoints_with_mms(frame, box)
                flat_keypoints = np.array(keypoints_frame).flatten()

                keypoints.append(flat_keypoints)
                frame_data.append((frame, keypoints_frame, dog_boxes, frame_index))

        frame_index += 1

    cap.release()

    if keypoints:
        input_tensor = torch.tensor(np.array(keypoints), dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            predicted_classes = torch.argmax(outputs, dim=1).tolist()
    else:
        predicted_classes = []

    return frame_data, predicted_classes

# 시각화 및 저장
def visualize_and_save_results(frame_data, predicted_classes, output_path="output_video.mp4"):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    behavior_counter = Counter()

    for i, (frame, keypoints, dog_boxes, frame_index) in enumerate(frame_data):
        if out is None:
            height, width, _ = frame.shape
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

        predicted_class = predicted_classes[i]
        behavior_counter[behavior_classes[predicted_class]] += 1

        # 바운딩 박스 그리기
        for (x1, y1, w, h) in dog_boxes:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

        # 키포인트 그리기
        for kp in keypoints:
            x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # 행동 예측 텍스트 추가
        cv2.putText(
            frame,
            f"Predicted: {behavior_classes[predicted_class]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    if out:
        out.release()

    # 행동 퍼센티지 계산
    total_frames = sum(behavior_counter.values())
    behavior_percentages = {behavior: (count / total_frames) * 100 for behavior, count in behavior_counter.items()}

    # 결과 출력
    print(f"결과 영상이 {output_path}에 저장되었습니다.")
    print("행동 분석 결과:")
    for behavior, percentage in behavior_percentages.items():
        print(f"{behavior}: {percentage:.2f}%")

# 실행 코드
if __name__ == "__main__":
    video_path = r"video_test\1.mp4"
    frame_data, predicted_classes = process_video(video_path)

    if frame_data and predicted_classes:
        visualize_and_save_results(frame_data, predicted_classes, output_path="output_video.mp4")
    else:
        print("키포인트를 추출하거나 행동을 예측하지 못했습니다.")
