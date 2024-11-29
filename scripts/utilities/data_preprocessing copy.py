import gdown
import zipfile
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 행동 이름 -> 숫자 레이블 매핑
behavior_classes = {
    'bodylower': 0,
    'bodyscratch': 1,
    'bodyshake': 2,
    'feetup': 3,
    'footup': 4,
    'heading': 5,
    'lying': 6,
    'mounting': 7,
    'sit': 8,
    'tailing': 9,
    'turn': 10,
    'walkrun': 11
}



def load_and_match_data(labeling_dir, frame_dir, output_csv):
    data_pairs = []

    # CSV 파일 생성
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["frame_number", "frame_path"] + [f"x{i}" if i % 2 == 1 else f"y{i//2}" for i in range(1, 31)]
        writer.writerow(header)

        for json_file in os.listdir(labeling_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(labeling_dir, json_file)

                # JSON 데이터 로드
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                except UnicodeDecodeError:
                    with open(json_path, 'r', encoding='utf-8-sig') as f:
                        label_data = json.load(f)

                print(f"JSON 파일 처리 중: {json_file}")

                # 이미지 폴더 매칭
                dog_name = json_file.split(".")[0]
                frames_path = os.path.join(frame_dir, dog_name)
                if not os.path.exists(frames_path):
                    print(f"이미지 폴더가 존재하지 않습니다: {frames_path}")
                    continue

                frame_files = sorted(os.listdir(frames_path))
                print(f"{dog_name} 폴더의 이미지 파일 수: {len(frame_files)}")

                for annotation in label_data.get("annotations", []):
                    frame_number = annotation.get("frame_number")
                    behavior = annotation.get
                    if frame_number is not None and frame_number < len(frame_files):
                        frame_path = os.path.join(frames_path, frame_files[frame_number])
                        keypoints = annotation.get("keypoints", {})
                        keypoints_flat = [
                            keypoints[str(i)]["x"] if keypoints.get(str(i)) else None
                            for i in range(1, 16)
                        ] + [
                            keypoints[str(i)]["y"] if keypoints.get(str(i)) else None
                            for i in range(1, 16)
                        ]
                        data_pairs.append((frame_path, label_data))
                        row = [frame_number, frame_path] + keypoints_flat
                        writer.writerow(row)

    print(f"CSV 파일 '{output_csv}' 생성 완료.")
    print(f"데이터 쌍 생성 완료: {len(data_pairs)}개 데이터")
    return data_pairs


if __name__ == "__main__":
    # 데이터 경로
    labeling_dir = 'data/train/taillow/labeling_taillow'
    frame_dir = 'data/train/taillow/frame_taillow'

    # CSV 출력 파일 경로
    os.makedirs('data/csv_file', exist_ok=True)  # CSV 파일 저장 경로 생성
    output_csv = 'data/csv_file/annotations_taillow.csv'

    # 데이터 로드 및 매칭 후 CSV 생성
    load_and_match_data(labeling_dir, frame_dir, output_csv)