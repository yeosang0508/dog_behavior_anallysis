import os
import json
import numpy as np
import pandas as pd
import csv

# 관절 이름 설정 (1부터 15까지)
joints_name = {
    1: 'Nose', 2: 'Forehead', 3: 'Mouth Corner', 4: 'Lower Lip', 5: 'Neck',
    6: 'Right Front Leg Start', 7: 'Left Front Leg Start', 8: 'Right Front Ankle', 9: 'Left Front Ankle',
    10: 'Right Thigh', 11: 'Left Thigh', 12: 'Right Rear Ankle', 13: 'Left Rear Ankle',
    14: 'Tail Start', 15: 'Tail End'
}

# 행동 라벨 정의
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

def visualize_keypoints_and_box(image, keypoints, bounding_box, save_path=None):
    """
    이미지에 키포인트와 바운딩 박스를 시각화합니다.
    이미지 크기에 맞게 바운딩 박스와 키포인트 좌표를 비례적으로 조정합니다.
    """
    image_height, image_width = image.shape[:2]  # 이미지 크기 확인

    # 바운딩 박스 그리기 (이미지 크기에 맞게 비례 조정)
    if bounding_box:
        x, y, w, h = map(int, [bounding_box['x'], bounding_box['y'], bounding_box['width'], bounding_box['height']])
        # 이미지 크기에 비례 맞추기
        x = int(x * image_width)
        y = int(y * image_height)
        w = int(w * image_width)
        h = int(h * image_height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 키포인트 그리기 (이미지 크기에 맞게 비례 조정)
    for i in range(1, 16):
        x = keypoints.get(str(i), {}).get("x")
        y = keypoints.get(str(i), {}).get("y")
        if x is not None and y is not None:
            x, y = int(x * image_width), int(y * image_height)  # 비례 맞추기
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, joints_name.get(i, f"Keypoint {i}"), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 이미지 저장 (옵션)
    if save_path:
        cv2.imwrite(save_path, image)

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

            # 디버깅용 출력
            print(f"frames_path: {frames_path}")
            if not os.path.exists(frames_path):
                print(f"경로가 존재하지 않습니다: {frames_path}")
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
                        # 여기서 실제 강아지만을 감지하는 모델을 사용하여 자르기
                        cropped_image = cv2.imread(frame_path)
                        if cropped_image is not None:
                            x, y, w, h = map(int, [bounding_box['x'], bounding_box['y'], bounding_box['width'], bounding_box['height']])
                            cropped_image = cropped_image[y:y + h, x:x + w]
                            cropped_path = os.path.join(video_cropped_folder, f"frame_{frame_number}.jpg")
                            cv2.imwrite(cropped_path, cropped_image)
                            detection_success = True

                            # 키포인트 시각화된 이미지 저장
                            vis_save_path = os.path.join(video_cropped_folder, f"vis_frame_{frame_number}.jpg")
                            visualize_keypoints_and_box(cropped_image, keypoints, bounding_box, save_path=vis_save_path)

                    # CSV 파일에 키포인트 정보 저장
                    keypoints_flat = [
                        keypoints[str(i)]["x"] if keypoints.get(str(i)) else None
                        for i in range(1, 16)
                    ] + [
                        keypoints[str(i)]["y"] if keypoints.get(str(i)) else None
                        for i in range(1, 16)
                    ]

                    # 수정된 부분: CSV 파일에 저장하는 row 수정
                    writer.writerow([frame_number, frame_path, cropped_path, detection_success, behavior_label] + keypoints_flat)

    print(f"CSV 파일 '{output_csv}' 생성 완료.")
    return data_pairs

# 메인 실행 함수
if __name__ == "__main__":
    # 경로 설정
    train_dir = r"C:\Users\admin\IdeaProjects\test\VSCode\data\train"
    csv_output = r"C:\Users\admin\IdeaProjects\test\VSCode\data\csv_file\annotations_bodylower.csv"
    cropped_output = r"C:\Users\admin\IdeaProjects\test\VSCode\data\cropped_bodylower"
    output_folder = r"C:\Users\admin\IdeaProjects\test\VSCode\data\visualizations"  # 시각화 이미지 저장 폴더

    # 행동 라벨 지정
    behavior_label = behavior_classes["bodylower"]

    # JSON 데이터를 읽고 CSV를 생성하며 이미지에 키포인트를 시각화
    load_and_match_data(
        labeling_dir=os.path.join(train_dir, "bodylower", "labeling_bodylower"),
        frame_dir=os.path.join(train_dir, "bodylower", "frame_bodylower"),
        output_csv=csv_output,
        cropped_folder=cropped_output,
        behavior_label=behavior_label
    )
