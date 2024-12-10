import cv2
import os
import numpy as np

# COCO 클래스 (강아지는 class 12번)
COCO_CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", 
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", 
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", 
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# MobileNet SSD 모델 로드
net = cv2.dnn.readNetFromCaffe('MobileNet/deploy.prototxt', 'MobileNet/mobilenet_iter_73000.caffemodel')

# FPS를 추출하는 함수
def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS 속성 가져오기
    cap.release()
    return fps

# 비디오에서 프레임을 추출하는 함수
def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 비디오 끝
        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)  # 프레임 저장
        print(f"Saved {frame_filename}")

    cap.release()
    print(f"총 {frame_count}개의 프레임이 추출되었습니다.")

# 강아지 감지 함수
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

# 바운딩 박스 정보 저장 함수
def save_bbox_info(bbox_list, output_folder):
    if bbox_list:
        with open(os.path.join(output_folder, "first_frame_bbox.txt"), 'w') as f:
            for box in bbox_list:
                f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
        print("바운딩 박스 정보 저장됨: first_frame_bbox.txt")

# 사용 예시
video_path = "video_test/21.mp4"  # 비디오 파일 경로
output_folder = "data/video1/21"  # 프레임이 저장될 폴더 경로

# FPS 확인
fps = get_video_fps(video_path)
if fps:
    print(f"비디오의 FPS: {fps}")

# 프레임 추출
extract_frames(video_path, output_folder)

# 첫 번째 프레임에 대해 강아지 감지 수행
frame_files = [f for f in os.listdir(output_folder) if f.endswith(".jpg")]
if frame_files:
    first_frame_path = os.path.join(output_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    # 첫 번째 프레임에서 강아지 감지
    dog_boxes = detect_dog_in_frame(first_frame)

    # 강아지 감지된 첫 번째 프레임 저장
    if dog_boxes:
        detected_frame_path = os.path.join(output_folder, "detected_first_frame.jpg")
        for (x1, y1, w, h) in dog_boxes:
            cv2.rectangle(first_frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.imwrite(detected_frame_path, first_frame)
        print(f"강아지 감지된 첫 번째 프레임 저장: {detected_frame_path}")

    # 첫 번째 프레임 바운딩 박스 정보 저장
    save_bbox_info(dog_boxes, output_folder)
