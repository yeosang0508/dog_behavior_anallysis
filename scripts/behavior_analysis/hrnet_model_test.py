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

print("HRNET_ROOT:", HRNET_ROOT)  # 디버그용 출력
print("sys.path:")
for path in sys.path:
    print(path)

# HRNet 관련 코드 import
from lib.models.cls_hrnet import HighResolutionNet
from lib.config import config as cfg
from lib.config import update_config

class HRNetModel:
    def __init__(self, model_path):
        # HRNet 설정 파일 경로
        config_file = os.path.join(HRNET_ROOT, "experiments", "cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml")
        
        # 설정 파일 경로 확인
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"설정 파일이 존재하지 않습니다. 경로를 확인하세요: {config_file}")
        else:
            print(f"설정 파일 확인됨: {config_file}")

        # 데이터 디렉토리 설정
        data_dir = os.path.abspath("C:/Users/admin/IdeaProjects/test/VSCode/data")

        # argparse를 사용해 args 객체 생성
        args = argparse.Namespace(
            cfg=config_file,
            modelDir="",       # 모델 디렉토리 정보
            logDir="",         # 로그 디렉토리 정보
            dataDir=data_dir,  # 데이터 디렉토리 경로
            testModel="",      # 테스트 모델 경로
            opts=[]            # 추가 설정 옵션
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
        self.model.to(self.device)  # 모델을 디바이스로 이동
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


# CSV 파일의 키포인트 업데이트 함수
def update_keypoints_from_csv(csv_path, output_csv, model):
    # CSV 파일 읽기
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")
    
    df = pd.read_csv(csv_path)

    # 키포인트 예측 결과를 저장할 리스트
    updated_rows = []

    for index, row in df.iterrows():
        frame_path = row["frame_path"]
        label = row["label"]  # 행동 레이블

        if not os.path.exists(frame_path):
            print(f"이미지 파일이 존재하지 않습니다: {frame_path}")
            continue

        # HRNet 모델을 사용해 키포인트 추출
        try:
            keypoints = model.predict_keypoints(frame_path)
            keypoints_flattened = keypoints.flatten().tolist()

            # 기존 데이터에 키포인트 추가
            updated_row = [row["frame_number"], frame_path] + keypoints_flattened + [label]
            updated_rows.append(updated_row)
        except Exception as e:
            print(f"키포인트 추출 중 오류 발생: {e} (frame: {frame_path})")

    # 업데이트된 데이터를 새로운 DataFrame으로 변환
    updated_columns = ["frame_number", "frame_path"] + [f"x{i},y{i}" for i in range(1, 16)] + ["label"]
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
    input_csv = "data/csv_file/annotations_combined.csv"  # 기존 CSV 파일
    output_csv = "data/csv_file/annotations_keypoints_updated.csv"  # 업데이트된 CSV 파일

    # CSV 파일에서 키포인트 업데이트 실행
    try:
        update_keypoints_from_csv(input_csv, output_csv, hrnet)
    except FileNotFoundError as e:
        print(e)
