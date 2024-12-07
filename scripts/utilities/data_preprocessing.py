import os
import sys
import cv2
import csv
import json
import numpy as np
import torch
import hydra
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
import matplotlib.pyplot as plt

# 현재 스크립트 디렉토리와 루트 디렉토리 설정
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 위치
project_root = os.path.abspath(os.path.join(current_dir, "../../"))  # 루트 디렉토리
sam2_path = os.path.join(project_root, "sam2")  # sam2 디렉토리 경로

# sys.path에 루트 및 sam2 경로 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"sys.path 설정: {sys.path[:5]}")  # 디버깅용

# Hydra 초기화 상태 확인 및 클리어
def reset_hydra():
    """
    Hydra가 이미 초기화된 상태라면 초기화를 클리어합니다.
    """
    if GlobalHydra.instance().is_initialized():
        initialize(config_path="sam2/sam2/configs", job_name="sam2", version_base="1.2")
        print("Hydra가 이미 초기화되어 있습니다. 초기화를 클리어합니다.")
        GlobalHydra.instance().clear()
    else:
        print("Hydra가 초기화되어 있지 않습니다.")

    try:
        cfg = compose(config_name="sam2_hiera_t.yaml")
        print("로드된 설정 내용:", OmegaConf.to_yaml(cfg))
    except Exception as e:
        print(f"설정 파일 로드 중 오류 발생: {e}")
        exit(1)

# Hydra 초기화
def initialize_hydra():
    """
    Hydra 초기화 상태를 클리어하고, 새로 초기화합니다.
    """
    reset_hydra()  # 기존 초기화 상태 클리어
    try:
        # YAML 파일의 절대 경로 설정
        config_path = os.path.abspath(os.path.join(project_root, "sam2/sam2/configs"))
        print(f"Hydra 설정 경로: {config_path}")

        print(f"Config path: {config_path}")
        print(f"YAML file path: {yaml_file}")
        print(f"Config path 존재 여부: {os.path.exists(config_path)}")
        print(f"YAML 파일 존재 여부: {os.path.exists(yaml_file)}")

        # Hydra 초기화
        hydra.initialize(config_path=config_path, version_base="1.2")
        print(f"Hydra 설정 모듈을 성공적으로 초기화했습니다. 설정 경로: {config_path}")
    except Exception as e:
        print(f"Hydra 설정 모듈 초기화 중 오류 발생: {e}")

# SAM 모델 초기화 및 로드 함수
def initialize_and_load_sam_model(cfg: DictConfig):
    """
    Hydra 설정을 받아 SAM 모델을 초기화하고 로드합니다.
    """
    # 모델 경로 확인 및 기본값 처리
    model_path = cfg.get("sam_model_path", None)
    if not model_path:
        raise ValueError("`sam_model_path`가 설정 파일에 없습니다. 설정 파일을 확인하세요.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # 모델 빌드
    sam_model = build_sam2(cfg, model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model.to(device)
    print("SAM 모델이 성공적으로 빌드되었습니다.")
    
    return SAM2ImagePredictor(sam_model)

# 강아지 영역 감지 및 시각화 함수
def detect_and_visualize(image_path, predictor, output_dir=None, visualize=False):
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
            with open(json_path, "r", encoding="utf-8") as f:
                annotations = json.load(f).get("annotations", [])
            
            folder_name = os.path.splitext(json_file)[0]
            frame_path = os.path.join(frame_dir, folder_name)
            
            for annotation in annotations:
                frame_number = annotation["frame_number"]
                keypoints = annotation.get("keypoints", {})
                frame_file = os.path.join(frame_path, f"frame_{frame_number}.jpg")
                
                cropped_regions, vis_image = detect_and_visualize(
                    frame_file, predictor, output_dir=vis_dir, visualize=True
                )
                
                for idx, (cropped_image, bbox) in enumerate(cropped_regions):
                    cropped_path = os.path.join(frame_path, f"cropped_{frame_number}_{idx}.jpg")
                    cv2.imwrite(cropped_path, cropped_image)
                    
                    keypoints_flat = [keypoints[str(i)][coord] for i in range(1, 16) for coord in ["x", "y"]]
                    writer.writerow([frame_number, cropped_path, bbox, keypoints_flat])

if __name__ == "__main__":
    reset_hydra()  # Hydra 초기화 상태 클리어
    initialize_hydra()  # Hydra 초기화 후 설정 경로 재구성
    cfg = compose(config_name="sam2/sam2_hiera_t.yaml")
    print("로드된 설정 내용:", cfg)

    # 경로 정의
    train_labeling_dir = r"C:/Users/admin/IdeaProjects/VSCode/data/train/bodylower/labeling_bodylower"
    train_frame_dir = r"C:/Users/admin/IdeaProjects/VSCode/data/train/bodylower/frame_bodylower"
    output_csv_path = r"C:/Users/admin/IdeaProjects/VSCode/data/csv_file/annotations_bodylower.csv"
    visualization_dir = r"C:/Users/admin/IdeaProjects/VSCode/visualizations"

    # SAM 모델 초기화 및 로드
    predictor = initialize_and_load_sam_model(cfg)


    yaml_path = "C:/Users/admin/IdeaProjects/VSCode/sam2/sam2/configs/sam2/sam2_hiera_t.yaml"
    print(f"YAML 파일 경로: {yaml_path}")
    print(f"YAML 파일 존재 여부: {os.path.exists(yaml_path)}")

    # 데이터 처리 및 시각화
    process_data_with_visualization(train_labeling_dir, train_frame_dir, output_csv_path, predictor, vis_dir=visualization_dir)
