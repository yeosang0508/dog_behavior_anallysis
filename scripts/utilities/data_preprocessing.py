import sys
import os
import gdown
import zipfile
import csv
import json
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

# SAMURAI 모델 경로
sam_model_path = r"C:\Users\admin\IdeaProjects\test\VSCode\sam2\samurai\sam2\checkpoints\sam2.1_hiera_base_plus.pt"

# Google Drive 데이터 파일 ID
bodylower_file_id = "11ArUhTu6Q8hNe2674d7_Ov3xlromti5h"

# 1. 데이터 다운로드 및 압축 해제
def download_and_extract(file_id, output_dir, zip_name):
    output_zip = os.path.join(output_dir, zip_name)
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {zip_name}...")
    gdown.download(url, output_zip, quiet=False)
    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"{zip_name} extracted to {output_dir}")

# 2. SAM 모델 로드
def load_sam_model(model_path):
    # SAM 모델 로드
    sam = sam_model_registry["vit_b"]()  # vit_b 사용 (체크포인트 유형에 맞춰 변경 가능)

    # 체크포인트 로드
    state_dict = torch.load(model_path, map_location="cpu")

    # SAMURAI 체크포인트의 "model" 키 필터링
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # 모델에 state_dict 적용
    missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")

    # SamPredictor 생성
    predictor = SamPredictor(sam)
    return predictor

# 3. 강아지 영역 감지 및 크롭
def detect_and_crop(image_path, predictor):
    image = cv2.imread(image_path)
    predictor.set_image(image)
    masks, _, _ = predictor.predict()
    cropped_regions = []
    for mask in masks:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))  # 마스크 바운딩 박스 계산
        cropped_image = image[y:y+h, x:x+w]
        cropped_regions.append(cropped_image)
    return cropped_regions

# 4. JSON과 이미지 매칭 후 CSV 생성
def process_data(labeling_dir, frame_dir, output_csv, predictor):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # CSV 헤더
        header = ["frame_number", "cropped_frame_path"] + [f"x{i}" if i % 2 else f"y{i//2}" for i in range(1, 31)]
        writer.writerow(header)

        for json_file in os.listdir(labeling_dir):
            if json_file.endswith(".json"):
                json_path = os.path.join(labeling_dir, json_file)
                with open(json_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f).get("annotations", [])
                dog_name = json_file.split(".")[0]
                frames_path = os.path.join(frame_dir, dog_name)
                if not os.path.exists(frames_path):
                    print(f"Frame folder does not exist: {frames_path}")
                    continue

                frame_files = sorted(os.listdir(frames_path))
                for annotation in annotations:
                    frame_number = annotation.get("frame_number")
                    if frame_number is not None and frame_number < len(frame_files):
                        frame_path = os.path.join(frames_path, frame_files[frame_number])
                        cropped_regions = detect_and_crop(frame_path, predictor)
                        for idx, cropped_image in enumerate(cropped_regions):
                            cropped_path = os.path.join(frames_path, f"cropped_{dog_name}_{frame_number}_{idx}.jpg")
                            cv2.imwrite(cropped_path, cropped_image)
                            keypoints = annotation.get("keypoints", {})
                            keypoints_flat = [
                                keypoints.get(str(i), {}).get("x") for i in range(1, 16)
                            ] + [
                                keypoints.get(str(i), {}).get("y") for i in range(1, 16)
                            ]
                            writer.writerow([frame_number, cropped_path] + keypoints_flat)
    print(f"CSV file created at: {output_csv}")

# 메인 실행
if __name__ == "__main__":
    # 데이터 경로 설정
    train_dir = "data/train"
    val_dir = "data/validation"
    csv_output = "data/csv_file/annotations.csv"

    # 데이터 다운로드 및 압축 해제
    download_and_extract(bodylower_file_id, train_dir, "bodylower.zip")
    download_and_extract(bodylower_file_id, val_dir, "bodylower.zip")

    # SAM 모델 로드
    sam_predictor = load_sam_model(sam_model_path)

    # JSON과 이미지 매칭 및 CSV 생성
    process_data(
        labeling_dir=os.path.join(train_dir, "bodylower/labeling_bodylower"),
        frame_dir=os.path.join(train_dir, "bodylower/frame_bodylower"),
        output_csv=csv_output,
        predictor=sam_predictor,
    )
