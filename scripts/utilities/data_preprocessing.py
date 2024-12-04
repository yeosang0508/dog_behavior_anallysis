import os
import cv2
import csv
import json
import numpy as np
import torch
from hydra import main
from omegaconf import DictConfig, OmegaConf
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

# Hydra 초기화 상태 확인 및 클리어
def reset_hydra():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

# SAM 모델 초기화 및 로드 함수
def initialize_and_load_sam_model(cfg: DictConfig):
    """
    Hydra 설정을 받아 SAM 모델을 초기화하고 로드합니다.
    """
    model_path = cfg.sam_model_path
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
    입력 이미지에서 강아지 영역을 감지하고 시각화합니다.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return [], image

    predictor.set_image(image)
    with torch.no_grad():
        masks, _, _ = predictor.predict(input_prompts=None)

    vis_image = image.copy()
    cropped_regions = []
    for mask in masks:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        cropped_image = image[y:y+h, x:x+w]
        cropped_regions.append((cropped_image, (x, y, w, h)))

        # 마스크 경계선 그리기
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (0, 255, 0), 2)  # 초록색

        # 경계 상자 그리기
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 파란색

    # 시각화 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"visualized_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to: {output_path}")

    # 시각화 플롯
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title("Object Detection Visualization")
        plt.axis("off")
        plt.show()

    return cropped_regions, vis_image

# JSON 데이터를 읽고 CSV 생성 및 시각화
def process_data_with_visualization(labeling_dir, frame_dir, output_csv, predictor, vis_dir=None):
    """
    JSON 데이터를 읽고 CSV로 저장하며, 감지된 이미지를 시각화합니다.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        
    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_number", "cropped_frame_path", "bounding_box", "keypoints"])
        
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

@main(config_path="C:/Users/admin/IdeaProjects/VSCode/sam2/sam2/configs/sam2.1", 
      config_name="sam2.1_hiera_t.yaml", 
      version_base=None)
def main(cfg: DictConfig):
    reset_hydra()  # Hydra 초기화 클리어

    # 디버깅: YAML 설정 로드 확인
    print("Loaded configuration:\n", OmegaConf.to_yaml(cfg))

    # 디버깅: 런타임 디렉토리 확인
    runtime_config = HydraConfig.get()
    print("Hydra Runtime Config:\n", runtime_config)
    
    # 경로 정의
    train_labeling_dir = r"C:\Users\admin\IdeaProjects\VSCode\data\train\bodylower\labeling_bodylower"
    train_frame_dir = r"C:\Users\admin\IdeaProjects\VSCode\data\train\bodylower\frame_bodylower"
    output_csv_path = "data/csv_file/annotations_bodylower.csv"
    visualization_dir = r"C:\Users\admin\IdeaProjects\VSCode\visualizations"

    # SAM 모델 초기화 및 로드
    predictor = initialize_and_load_sam_model(cfg)

    # 데이터 처리 및 시각화
    process_data_with_visualization(train_labeling_dir, train_frame_dir, output_csv_path, predictor, vis_dir=visualization_dir)

if __name__ == "__main__":
    reset_hydra()  # 실행 전에 Hydra 초기화 클리어
    main()
