import gdown
import zipfile
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def download_file(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f"Downloading {output}...")
    gdown.download(url, output, quiet=False)
    print(f"{output} downloaded.")

def extract_zip(file_name, extract_to):
    if os.path.exists(file_name):
        print(f"Extracting {file_name}...")
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"{file_name} extracted to {extract_to}.")
    else:
        print(f"{file_name} not found.")

# Google Drive 에서 가져온 파일 ID
sit_file_id = '11BjnynF280Qfs2eDOQiKIYkHe7o6O04o' 


# 데이터 폴더 생성 
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

# 파일 다운로드 
download_file(sit_file_id, 'data/train/sit.zip')
download_file(sit_file_id, 'data/validation/sit.zip')


# 압축 해제 경로 설정
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

# 압축 해제
extract_zip('data/train/sit.zip', 'data/train')
extract_zip('data/validation/sit.zip', 'data/validation')


def load_and_match_data(labeling_dir, frame_dir, output_csv):
    data_pairs = []

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

                # 이미지 폴더 매칭 로직
                if ".mp4" in json_file:
                    dog_name = json_file.replace(".mp4.json", "")
                else:
                    dog_name = json_file.replace(".json", "")

                # 이미지 폴더 이름에서 `.mp4` 제거
                frames_path = os.path.normpath(os.path.join(frame_dir, dog_name.replace(".mp4", "")))

                # 디버깅용 출력
                print(f"생성된 frames_path: {frames_path}")
                if not os.path.exists(frames_path):
                    print(f"이미지 폴더가 존재하지 않습니다: {frames_path}")
                    continue

                frame_files = sorted(os.listdir(frames_path))
                print(f"{dog_name} 폴더의 이미지 파일 수: {len(frame_files)}")

                for annotation in label_data.get("annotations", []):
                    frame_number = annotation.get("frame_number")
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
    labeling_dir = 'data/train/sit/labeling_sit'
    frame_dir = 'data/train/sit/frame_sit'

    # CSV 출력 파일 경로
    os.makedirs('data/csv_file', exist_ok=True)  # CSV 파일 저장 경로 생성
    output_csv = 'data/csv_file/annotations_sit.csv'

    # 데이터 로드 및 매칭 후 CSV 생성
    load_and_match_data(labeling_dir, frame_dir, output_csv)