import gdown
import zipfile
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import matplotlib.pyplot as plt
import cv2

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
bodyshake_file_id = '1gzM1d-mQkJB6l4HD-Xq31i7JAfyK2mAR'

# 데이터 폴더 생성
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

# 파일 다운로드
download_file(bodyshake_file_id, 'data/train/bodyshake.zip')
download_file(bodyshake_file_id, 'data/validation/bodyshake.zip')

# 압축 해제 경로 설정
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

# 압축 해제
extract_zip('data/train/bodyshake.zip', 'data/train')
extract_zip('data/validation/bodyshake.zip', 'data/validation')


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
                    if frame_number is not None and frame_number < len(frame_files):
                        frame_path = os.path.join(frames_path, frame_files[frame_number])
                        keypoints = annotation.get("keypoints", {})

                        # 좌표 유효성 확인 및 처리
                        keypoints_flat = []
                        for i in range(1, 16):
                            x = keypoints.get(str(i), {}).get("x")
                            y = keypoints.get(str(i), {}).get("y")
                            if x is None or y is None or x < 0 or y < 0:  # 누락값 처리 및 범위 검증
                                x, y = 0, 0  # 누락값을 0으로 처리
                            keypoints_flat.extend([x, y])

                        # 데이터 추가
                        data_pairs.append((frame_path, label_data))
                        row = [frame_number, frame_path] + keypoints_flat
                        writer.writerow(row)

    print(f"CSV 파일 '{output_csv}' 생성 완료.")
    print(f"데이터 쌍 생성 완료: {len(data_pairs)}개 데이터")
    return data_pairs


def validate_data(csv_path):
    # 데이터 로드
    import pandas as pd
    df = pd.read_csv(csv_path)

    # 누락값 확인
    print("\n누락값 확인:")
    print(df.isnull().sum())

    # 좌표 값의 유효성 확인
    print("\n좌표 유효성 확인:")
    invalid_coords = df[(df.filter(like='x') < 0).any(axis=1) | (df.filter(like='y') < 0).any(axis=1)]
    print(f"잘못된 좌표가 포함된 행 개수: {len(invalid_coords)}")

    # 좌표 정규화
    print("\n좌표 정규화:")
    image_width, image_height = 1080, 1920  # 예제 이미지 크기
    coord_columns = [col for col in df.columns if col.startswith('x') or col.startswith('y')]
    df[coord_columns] = df[coord_columns].apply(lambda x: x / (image_width if 'x' in x.name else image_height))
    print("정규화 완료")

    # 중복 확인
    print("\n중복 데이터 확인:")
    duplicates = df[df.duplicated()]
    print(f"중복된 행 개수: {len(duplicates)}")

    # 데이터 균형 확인
    print("\n데이터 균형 확인:")
    print(df['frame_path'].value_counts())

    # 업데이트된 데이터 저장
    df.to_csv(csv_path, index=False)
    print(f"검증 후 CSV 저장 완료: {csv_path}")


if __name__ == "__main__":
    # 데이터 경로
    labeling_dir = 'data/train/bodyshake/labeling_bodyshake'
    frame_dir = 'data/train/bodyshake/frame_bodyshake'

    # CSV 출력 파일 경로
    os.makedirs('data/csv_file', exist_ok=True)  # CSV 파일 저장 경로 생성
    output_csv = 'data/csv_file/annotations_bodyshake.csv'

    # 데이터 로드 및 매칭 후 CSV 생성
    load_and_match_data(labeling_dir, frame_dir, output_csv)

    # 데이터 검증 및 정리
    validate_data(output_csv)

   # CSV 파일 로드
    csv_path = "data/csv_file/annotations_bodyshake.csv"
    df = pd.read_csv(csv_path)

    # 샘플 데이터 시각화
    sample_frame = df.iloc[0]  # 첫 번째 샘플 데이터 선택
    image_path = sample_frame["frame_path"]  # 이미지 경로
    keypoints = sample_frame[2:].values.reshape(-1, 2)  # x, y로 변환

    # 이미지 읽기 및 시각화
    image = cv2.imread(image_path)
    for (x, y) in keypoints:
        if not pd.isnull(x) and not pd.isnull(y):  # 유효 좌표만 시각화
            cv2.circle(image, (int(x * 1080), int(y * 1920)), radius=5, color=(0, 255, 0), thickness=-1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")  # 축 숨기기
    plt.show()