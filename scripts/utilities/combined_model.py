#HRNet과 KeypointRCNN을 결합

import os
import sys
import argparse
import torch
from torchvision.transforms import transforms
import cv2
import numpy as np
import pandas as pd  # CSV 작업에 사용

# CPU 병렬 처리 제한 (코어 수 제한)
torch.set_num_threads(4)  # CPU 코어 수를 4개로 제한

# HRNet-Image-Classification 루트 경로를 sys.path에 추가
HRNET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../HRNet-Image-Classification"))
sys.path.append(HRNET_ROOT)

# HRNet 관련 코드 import
from lib.models.cls_hrnet import HighResolutionNet
from lib.config import config as cfg
from lib.config import update_config

# KeypointRCNN Import
from KeypointRCNN import predict_keypoints_rcnn

class HRNetModel:
    def __init__(self, model_path):
        # HRNet 설정 파일 경로
        config_file = os.path.join(HRNET_ROOT, "experiments", "cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        
        # 설정 파일 경로 확인
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"설정 파일이 존재하지 않습니다. 경로를 확인하세요: {config_file}")
        
        # 데이터 디렉토리 설정
        data_dir = os.path.abspath("C:/Users/admin/IdeaProjects/test/VSCode/data")

        # argparse를 사용해 args 객체 생성
        args = argparse.Namespace(
            cfg=config_file,
            modelDir="",
            logDir="",
            dataDir=data_dir,
            testModel="",
            opts=[]
        )
        
        # HRNet 설정 업데이트
        update_config(cfg, args)

        # 디바이스 설정 (GPU 우선)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # HRNet 모델 초기화
        self.model = HighResolutionNet(cfg)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 가중치 파일이 존재하지 않습니다. 경로를 확인하세요: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 이미지 전처리 설정
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_keypoints(self, image_path):
        # 이미지 로드 및 전처리
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        # 모델 추론
        with torch.no_grad():
            output = self.model(input_tensor)

        # 키포인트 추출
        keypoints = output.cpu().numpy()
        return keypoints

# CSV 파일의 키포인트 업데이트 함수 (HRNet 및 KeypointRCNN 모두 사용)
def update_keypoints_from_csv(csv_path, output_csv, hrnet_model):
    # CSV 파일 읽기
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")
    
    df = pd.read_csv(csv_path)

    # 키포인트 예측 결과를 저장할 리스트
    updated_rows = []

    for index, row in df.iterrows():
        frame_path = row["frame_path"]
        label = row["label"]

        if not os.path.exists(frame_path):
            print(f"이미지 파일이 존재하지 않습니다: {frame_path}")
            continue

        # HRNet을 사용하여 키포인트 추출
        try:
            keypoints_hrnet = hrnet_model.predict_keypoints(frame_path)
            keypoints_hrnet_flattened = keypoints_hrnet.flatten().tolist()
        except Exception as e:
            print(f"HRNet 키포인트 추출 중 오류 발생: {e} (frame: {frame_path})")
            keypoints_hrnet_flattened = [None] * 30  # HRNet 키포인트의 예상 크기

        # KeypointRCNN을 사용하여 키포인트 추출
        try:
            keypoints_rcnn, _ = predict_keypoints_rcnn(frame_path)
            keypoints_rcnn_flattened = keypoints_rcnn.flatten().tolist()
        except Exception as e:
            print(f"KeypointRCNN 키포인트 추출 중 오류 발생: {e} (frame: {frame_path})")
            keypoints_rcnn_flattened = [None] * 30  # KeypointRCNN 키포인트의 예상 크기

        # 기존 데이터에 HRNet과 KeypointRCNN의 키포인트 모두 추가
        updated_row = [row["frame_number"], frame_path] + keypoints_hrnet_flattened + keypoints_rcnn_flattened + [label]
        updated_rows.append(updated_row)

    # 업데이트된 데이터를 새로운 DataFrame으로 변환
    updated_columns = ["frame_number", "frame_path"] + \
                      [f"hrnet_x{i},hrnet_y{i}" for i in range(1, 16)] + \
                      [f"rcnn_x{i},rcnn_y{i}" for i in range(1, 16)] + ["label"]
    updated_df = pd.DataFrame(updated_rows, columns=updated_columns)

    # 결과를 새로운 CSV 파일로 저장
    updated_df.to_csv(output_csv, index=False)
    print(f"업데이트된 CSV 저장 완료: {output_csv}")

# 테스트 코드
if __name__ == "__main__":
    # 모델 가중치 파일 경로
    model_path = os.path.join(HRNET_ROOT, "hrnetv2_w48_imagenet_pretrained.pth")

    # HRNet 모델 초기화
    hrnet = HRNetModel(model_path)

    # CSV 파일 경로
    input_csv = "data/csv_file/annotations_combined.csv"
    output_csv = "data/csv_file/annotations_keypoints_updated.csv"

    # CSV 파일에서 키포인트 업데이트 실행 (HRNet 및 KeypointRCNN 모두 적용)
    try:
        update_keypoints_from_csv(input_csv, output_csv, hrnet)
    except FileNotFoundError as e:
        print(e)