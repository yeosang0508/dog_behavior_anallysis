import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from config.config import Config
from stgcn_model import STGCN

# ----- 모델 로드 -----
def load_trained_model(config):
    checkpoint_path = os.path.join("models", "stgcn_behavior.pth")

    print(f"[INFO] 모델 로드: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[ERROR] 모델 파일을 찾을 수 없습니다: {checkpoint_path}")

    # 모델 초기화
    model = STGCN(
        in_channels=2,
        num_joints=config.num_joints,
        num_classes=config.num_classes,
        num_frames=30
    ).to(config.device)

    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("[INFO] 모델 로드 완료")
    return model

# ----- 스켈레톤 데이터 추출 -----
def extract_skeleton_from_video(video_path, config):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    skeleton_data = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 스켈레톤 추출
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        else:
            keypoints = [(0, 0)] * config.num_joints

        skeleton_data.append(keypoints)
        frame_count += 1

    cap.release()
    pose.close()

    print(f"[INFO] 스켈레톤 데이터 추출 완료. 총 {frame_count} 프레임 처리됨.")
    return np.array(skeleton_data, dtype=np.float32)

# ----- 행동 지속 시간 계산 -----
def calculate_behavior_duration(predictions, frame_rate=30):
    unique, counts = np.unique(predictions, return_counts=True)
    behavior_summary = dict(zip(unique, counts))

    print("\n[INFO] 행동 지속 시간 계산 결과:")
    for behavior_id, frame_count in behavior_summary.items():
        duration = frame_count / frame_rate  # 초 단위 지속 시간
        print(f" - 행동 {behavior_id}: {frame_count} 프레임 ({duration:.2f}초)")

    return behavior_summary

# ----- 행동 분석 -----
def test_model_with_video(video_path):
    config = Config()

    # 모델 로드
    model = load_trained_model(config)

    # 스켈레톤 데이터 추출
    skeleton_data = extract_skeleton_from_video(video_path, config)
    if len(skeleton_data) == 0:
        raise ValueError("[ERROR] 스켈레톤 데이터가 없습니다.")

    # 데이터 변환
    skeleton_tensor = torch.tensor(skeleton_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(config.device)

    # 모델 추론
    with torch.no_grad():
        outputs = model(skeleton_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    print(f"[INFO] 행동 분석 완료. 예측 결과: {predictions}")
    return predictions

# ----- 행동 분포 시각화 -----
def visualize_behavior_distribution(behavior_summary):
    labels = [f"행동 {key}" for key in behavior_summary.keys()]
    sizes = list(behavior_summary.values())

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("행동 분포")
    plt.axis('equal')
    plt.show()

# ----- 주요 행동 반환 -----
def get_primary_behavior(behavior_summary):
    primary_behavior = max(behavior_summary, key=behavior_summary.get)
    print(f"[INFO] 주요 행동: {primary_behavior} ({behavior_summary[primary_behavior]} 프레임)")
    return primary_behavior

# ----- 행동 전환 탐지 -----
def detect_behavior_transitions(predictions):
    transitions = []
    current_behavior = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_behavior:
            transitions.append((current_behavior, start_frame, i - 1))
            current_behavior = predictions[i]
            start_frame = i

    transitions.append((current_behavior, start_frame, len(predictions) - 1))
    return transitions
