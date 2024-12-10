import gdown
import zipfile
import csv
import os
import json

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
walkrun_file_id = '1YyPBapv1Yo9n2ZlHC98_Jh_hP7JqQ2zI' 

# 데이터 폴더 생성 
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

# 파일 다운로드 
download_file(walkrun_file_id, 'data/train/walkrun.zip')
download_file(walkrun_file_id, 'data/validation/walkrun.zip')

# 압축 해제 경로 설정
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

# 압축 해제
extract_zip('data/train/walkrun.zip', 'data/train')
extract_zip('data/validation/walkrun.zip', 'data/validation')

def load_and_match_data(labeling_dir, output_csv):
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["frame_number"] + [f"x{i}" if i % 2 == 1 else f"y{i//2}" for i in range(1, 31)] + ["label"]
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

                # `file_video`에서 행동 이름 추출
                file_video = label_data.get("file_video", "")
                behavior_name = None
                if file_video:
                    behavior_name = file_video.split("/")[-1].split("-")[1]  # 행동 이름 추출
                    print(f"추출된 행동 이름: {behavior_name}")

                # 라벨 찾기
                label = behavior_classes.get(behavior_name, -1)  # 기본값 -1

                for annotation in label_data.get("annotations", []):
                    frame_number = annotation.get("frame_number")
                    keypoints = annotation.get("keypoints", {})

                    keypoints_flat = [
                        keypoints[str(i)]["x"] if keypoints.get(str(i)) else None
                        for i in range(1, 16)
                    ] + [
                        keypoints[str(i)]["y"] if keypoints.get(str(i)) else None
                        for i in range(1, 16)
                    ]
                    row = [frame_number] + keypoints_flat + [label]
                    writer.writerow(row)

if __name__ == "__main__":
    # 데이터 경로
    labeling_dir = 'data/train/walkrun/labeling_walkrun'

    # CSV 출력 파일 경로
    os.makedirs('data/csv_file', exist_ok=True)  # CSV 파일 저장 경로 생성
    output_csv = 'data/csv_file/walkrun.csv'

    # 데이터 로드 및 매칭 후 CSV 생성
    load_and_match_data(labeling_dir, output_csv)
