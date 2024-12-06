import os
import sys
import csv
import json
import cv2
import numpy as np
import torch
from models.experimental import attempt_load  # 모델 로드 함수
from utils.general import non_max_suppression  # NMS 함수
from utils.datasets import letterbox  # 이미지 전처리 함수

# YOLOv7 디렉토리를 Python 경로에 추가
yolov7_path = r"C:\Users\admin\IdeaProjects\test\VSCode\yolov7"
if yolov7_path not in sys.path:
    sys.path.append(yolov7_path)

# COCO 클래스 정의
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# 행동 라벨 정의 (13개)
behavior_classes = {
    "bodylower": 0,
    "bodyscratch": 1,
    "bodyshake": 2,
    "feetup": 3,
    "footup": 4,
    "heading": 5,
    "lying": 6,
    "mounting": 7,
    "sit": 8,
    "tailing": 9,
    "taillow": 10,
    "turn": 11,
    "walkrun": 12
}

# YOLOv7 모델 로드 함수
def load_yolo_model():
    weights_path = r"C:\Users\admin\IdeaProjects\test\VSCode\yolov7\weights\yolov7-tiny.pt"

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"가중치 파일이 경로에 없습니다: {weights_path}")

    try:
        model = attempt_load(weights_path, map_location=torch.device('cpu'))
        model.eval()
        print(f"YOLOv7 모델 로드 완료: {weights_path}")
        return model
    except Exception as e:
        raise RuntimeError(f"YOLOv7 모델 로드 중 오류 발생: {e}")

# 키포인트와 바운딩 박스를 시각화
def visualize_keypoints_and_box(image, keypoints, bounding_box, save_path=None):
    if bounding_box:
        x, y, w, h = map(int, [bounding_box['x'], bounding_box['y'], bounding_box['width'], bounding_box['height']])
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색 박스

    for _, point in keypoints.items():
        if point:
            cv2.circle(image, (int(point['x']), int(point['y'])), 5, (0, 255, 0), -1)  # 초록색 점

    if save_path:
        cv2.imwrite(save_path, image)
    return image

# JSON 데이터와 이미지 매칭 및 CSV 생성
def load_and_match_data(labeling_dir, frame_dir, output_csv, cropped_folder, behavior_label): 
    os.makedirs(cropped_folder, exist_ok=True)
    data_pairs = []

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["frame_number", "frame_path", "cropped_path", "detection_success", "label"] + [f"x{i}" if i % 2 == 1 else f"y{i//2}" for i in range(1, 31)]
        writer.writerow(header)

        for json_file in os.listdir(labeling_dir):
            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(labeling_dir, json_file)

            # JSON 데이터 로드
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
            except UnicodeDecodeError:
                with open(json_path, 'r', encoding='utf-8-sig') as f:
                    label_data = json.load(f)

            print(f"JSON 파일 처리 중: {json_file}")

            dog_name = json_file.replace(".mp4.json", "") if ".mp4" in json_file else json_file.replace(".json", "")
            frames_path = os.path.normpath(os.path.join(frame_dir, dog_name.replace(".mp4", "")))

            if not os.path.exists(frames_path):
                print(f"이미지 폴더가 존재하지 않습니다: {frames_path}")
                continue

            video_cropped_folder = os.path.join(cropped_folder, dog_name)
            os.makedirs(video_cropped_folder, exist_ok=True)

            frame_files = sorted(os.listdir(frames_path))
            print(f"{dog_name} 폴더의 이미지 파일 수: {len(frame_files)}")

            for annotation in label_data.get("annotations", []):
                frame_number = annotation.get("frame_number")
                if frame_number is not None and frame_number < len(frame_files):
                    frame_path = os.path.join(frames_path, frame_files[frame_number])
                    keypoints = annotation.get("keypoints", {})
                    bounding_box = annotation.get("bounding_box")

                    detection_success = False
                    cropped_path = None

                    if bounding_box:
                        cropped_image = cv2.imread(frame_path)
                        if cropped_image is not None:
                            x, y, w, h = map(int, [bounding_box['x'], bounding_box['y'], bounding_box['width'], bounding_box['height']])
                            cropped_image = cropped_image[y:y + h, x:x + w]
                            cropped_path = os.path.join(video_cropped_folder, f"frame_{frame_number}.jpg")
                            cv2.imwrite(cropped_path, cropped_image)
                            detection_success = True

                            vis_save_path = os.path.join(video_cropped_folder, f"vis_frame_{frame_number}.jpg")
                            visualize_keypoints_and_box(cropped_image, keypoints, bounding_box, save_path=vis_save_path)

                    keypoints_flat = [
                        keypoints[str(i)]["x"] if keypoints.get(str(i)) else None
                        for i in range(1, 16)
                    ] + [
                        keypoints[str(i)]["y"] if keypoints.get(str(i)) else None
                        for i in range(1, 16)
                    ]

                    writer.writerow([frame_number, frame_path, cropped_path, detection_success, behavior_label] + keypoints_flat)

    print(f"CSV 파일 '{output_csv}' 생성 완료.")
    return data_pairs

# 메인 실행 함수
if __name__ == "__main__":
    train_dir = r"C:\Users\admin\IdeaProjects\test\VSCode\data\train"
    csv_output = r"C:\Users\admin\IdeaProjects\test\VSCode\data\csv_file\annotations_sit.csv"
    cropped_output = r"C:\Users\admin\IdeaProjects\test\VSCode\data\cropped_sit"

    behavior_label = behavior_classes["sit"]

    load_and_match_data(
        labeling_dir=os.path.join(train_dir, "sit", "labeling_sit"),
        frame_dir=os.path.join(train_dir, "sit", "frame_sit"),
        output_csv=csv_output,
        cropped_folder=cropped_output,
        behavior_label=behavior_label
    )