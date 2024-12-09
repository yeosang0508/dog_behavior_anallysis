import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from keypointModel import KeypointModel  # 훈련된 KeypointModel 클래스
from samurai import SamuraiTracker  # SAMURAI 객체 감지

# GPU/CPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 행동 클래스 정의
behavior_classes = {
    0: "bodylower",
    1: "bodyscratch",
    2: "bodyshake",
    3: "feetup",
    4: "footup",
    5: "heading",
    6: "lying",
    7: "mounting",
    8: "sit",
    9: "tailing",
    10: "taillow",
    11: "turn",
    12: "walkrun"
}

class DogBehaviorDetection:
    def __init__(self, model_path, behavior_classes, output_path):
        self.tracker = SamuraiTracker()  # SAMURAI 객체 감지 초기화
        self.model = torch.load(model_path).to(device)  # 사전 훈련된 모델 로드
        self.model.eval()  # 평가 모드로 설정
        self.behavior_classes = behavior_classes
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        all_results = []  # 프레임별 결과 저장

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # 객체 감지 및 추적 수행
            detection_result = self.tracker.track(frame)
            if detection_result["detection_success"]:
                cropped_frame = self.crop_frame(frame, detection_result["bbox"])
                keypoints = detection_result["keypoints"]

                # 행동 분류 수행
                behavior_class = self.classify_behavior(cropped_frame, keypoints)
                behavior_label = self.behavior_classes[behavior_class]

                # 결과 저장
                all_results.append({
                    "frame_id": frame_id,
                    "behavior_class": behavior_class,
                    "behavior_label": behavior_label,
                    "bbox": detection_result["bbox"],
                    "keypoints": keypoints
                })
        
        cap.release()
        self.save_results(all_results, video_path)
        print(f"Processing complete. Results saved to {self.output_path}")

    def classify_behavior(self, cropped_frame, keypoints):
        # 행동 분류 모델 입력 준비
        input_data = self.prepare_input(keypoints)
        with torch.no_grad():
            input_data = input_data.to(device)
            output = self.model(input_data)
            behavior_class = torch.argmax(output, dim=1).item()
        return behavior_class

    def prepare_input(self, keypoints):
        """
        행동 분류 모델 입력 데이터 준비
        """
        keypoints_array = np.array(keypoints).flatten().astype(np.float32)
        input_data = torch.tensor(keypoints_array).unsqueeze(0)  # 배치 차원 추가
        return input_data

    def crop_frame(self, frame, bbox):
        """
        감지된 객체를 기준으로 프레임 자르기
        """
        x, y, w, h = bbox
        return frame[int(y):int(y+h), int(x):int(x+w)]

    def save_results(self, results, video_path):
        """
        결과를 CSV로 저장
        """
        output_file = os.path.join(self.output_path, os.path.basename(video_path) + "_results.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)

# 실행
if __name__ == "__main__":
    # 결과 저장 경로
    output_path = "output_results/"
    os.makedirs(output_path, exist_ok=True)

    # 객체 감지 및 행동 분류 수행
    detector = DogBehaviorDetection(
        model_path="best_model.pth",  # 사전 훈련된 모델 경로
        behavior_classes=behavior_classes,
        output_path=output_path
    )

    # 테스트 비디오
    test_video_path = r"video_test\4.mp4"
    detector.process_video(test_video_path)
