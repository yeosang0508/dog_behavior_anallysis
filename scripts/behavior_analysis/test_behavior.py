import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import torch
import numpy as np
import mediapipe as mp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import BehaviorDataset
from model import STGCN
from config.config import Config
from matplotlib import font_manager, rc
# 행동 분석 결과 개선
from collections import Counter
from matplotlib import rcParams


# ----- 스켈레톤 데이터 추출 -----
def extract_skeleton_from_video(video_path, config, output_path="output_video.mp4"):
    """
    영상을 처리하여 각 프레임의 스켈레톤 데이터를 추출.
    MediaPipe를 사용하여 관절 좌표 추출.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    skeleton_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe로 스켈레톤 추출
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # 스켈레톤 좌표 추출
        keypoints = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y) for lm in landmarks]
        else:
            keypoints = [(0.0, 0.0)] * config.num_joints

        # 관절 수 조정
        if len(keypoints) != config.num_joints:
            keypoints = keypoints[:config.num_joints]
            keypoints += [(0.0, 0.0)] * (config.num_joints - len(keypoints))

        skeleton_data.append(keypoints)

        # 시각화 및 비디오 저장
        frame_with_skeleton = visualize_skeleton(frame, keypoints, config)
        out.write(frame_with_skeleton)
        frame_count += 1


    cap.release()
    out.release()
    pose.close()

    print(f"Processed {frame_count} frames. 관절 선이 그려진 영상이 '{output_path}'로 저장되었습니다.")
    return np.array(skeleton_data, dtype=np.float32)


def visualize_skeleton(image, keypoints, config):
    """
    스켈레톤 데이터를 프레임 위에 시각화하며, 관절 연결선도 그린다.
    """
    # 관절 표시
    for x, y in keypoints:
        if x is not None and y is not None:
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)

    # 관절 연결 그리기
    for (start, end) in config.joint_pair:
        if keypoints[start][0] is not None and keypoints[end][0] is not None:
            x1, y1 = keypoints[start]
            x2, y2 = keypoints[end]
            color = config.joint_colors.get(start, (0, 255, 0))
            cv2.line(
                image,
                (int(x1 * image.shape[1]), int(y1 * image.shape[0])),
                (int(x2 * image.shape[1]), int(y2 * image.shape[0])),
                color=color,
                thickness=2
            )
    return image


# ----- 행동 해석 -----
def interpret_behavior(predictions, config):
    """
    모델 예측 결과를 해석하여 행동을 한글로 출력.
    """
    if len(predictions) == 1:  # predictions가 단일 클래스만 포함할 경우
        behavior_name = config.behavior_classes.get(predictions[0], "알 수 없음")
        print(f"\n행동 분석 결과: {behavior_name} (단일 예측)")
        return

    unique, counts = np.unique(predictions, return_counts=True)
    behavior_summary = dict(zip(unique, counts))

    print("\n행동 분석 결과:")
    total_frames = sum(counts)
    for behavior_id, count in behavior_summary.items():
        behavior_name = config.behavior_classes.get(behavior_id, "알 수 없음")
        percentage = (count / total_frames) * 100
        print(f"- {behavior_name}: {count} 프레임 ({percentage:.1f}%)")

    # 주요 행동 출력
    primary_behavior = max(behavior_summary, key=behavior_summary.get)
    primary_behavior_name = config.behavior_classes.get(primary_behavior, "알 수 없음")
    print(f"\n주요 행동: {primary_behavior_name} ({behavior_summary[primary_behavior]} 프레임)")

    visualize_behavior_distribution(behavior_summary, config)

def set_korean_font():
    """
    Matplotlib에 한글 글꼴을 설정합니다.
    """
    font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows의 맑은 고딕 경로
    if not os.path.exists(font_path):
        print("한글 글꼴 파일을 찾을 수 없습니다. 적절한 경로를 설정하세요.")
        return
    
    font = font_manager.FontProperties(fname=font_path).get_name()
    rcParams['font.family'] = font
    rcParams['axes.unicode_minus'] = False  # 마이너스 기호 문제 해결
    print(f"설정된 한글 글꼴: {font}")

# 한글 글꼴 설정 호출
set_korean_font()

def visualize_behavior_distribution(behavior_summary, config):
    """
    행동 분포를 시각화 (파이차트).
    """
    labels = [config.behavior_classes.get(behavior_id, "알 수 없음") for behavior_id in behavior_summary.keys()]
    sizes = behavior_summary.values()

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("행동 분포")
    plt.axis('equal')
    plt.show()


# ----- 모델 테스트 ----- 
def test_model_with_video(video_path):
    config = Config()

    # 스켈레톤 데이터 추출
    skeleton_data = extract_skeleton_from_video(video_path, config)
    print(f"Skeleton data shape: {skeleton_data.shape}")

    if len(skeleton_data) == 0:
        print("영상에 관절 데이터가 감지되지 않았습니다. 다른 영상을 시도하세요.")
        return

    # 데이터 변환
    skeleton_tensor = torch.tensor(skeleton_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    print(f"Skeleton tensor shape: {skeleton_tensor.shape}")

    # 모델 생성
    model = STGCN(
        in_channels=2,
        num_joints=config.num_joints,
        num_classes=config.num_classes,
        num_frames=skeleton_tensor.shape[2]
    ).to(config.device)

    adjacency_matrix = torch.eye(config.num_joints).to(config.device)

    try:
        outputs = model(skeleton_tensor, adjacency_matrix)
        print(f"Model output shape: {outputs.shape}") #모델 출력의 차원 확인 
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        print(f"Predictions shape: {predictions.shape}, Predictions: {predictions}")

        interpret_behavior(predictions, config)
        print(f"Predicted behaviors for all frames: {predictions}")

    except Exception as e:
        print(f"Model forward pass failed: {e}")

    # 첫 프레임 시각화
    frame_idx = 0
    keypoints = skeleton_data[frame_idx]
    plt.scatter(*zip(*keypoints))
    plt.title("Skeleton Keypoints for Frame 0")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().invert_yaxis()
    plt.show()


# ----- 실행 -----
if __name__ == "__main__":
    config = Config()
    test_model_with_video(config.video_path)
