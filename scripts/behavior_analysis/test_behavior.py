import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import BehaviorDataset
from stgcn_model import STGCN
from config.config import Config
from matplotlib import font_manager, rc
# 행동 분석 결과 개선
from collections import Counter
from matplotlib import rcParams

config = Config()
config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

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


# 행동 분포 시각화 함수
def visualize_behavior_distribution(behavior_summary, config):
    labels = [config.behavior_classes.get(behavior_id, "알 수 없음") for behavior_id in behavior_summary.keys()]
    sizes = list(behavior_summary.values())

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("행동 분포")
    plt.axis('equal')
    plt.show()

# 행동 해석 함수
def interpret_behavior(predictions, config, frame_rate=30):
    unique, counts = np.unique(predictions, return_counts=True)
    behavior_summary = dict(zip(unique, counts))

    print("\n행동 분석 결과:")
    total_frames = len(predictions)
    for behavior_id, count in behavior_summary.items():
        behavior_name = config.behavior_classes.get(behavior_id, "알 수 없음")
        duration = count / frame_rate  # 초 단위 지속 시간
        percentage = (count / total_frames) * 100
        print(f"- {behavior_name}: {count} 프레임 ({percentage:.1f}%, 약 {duration:.1f}초)")

    primary_behavior = max(behavior_summary, key=behavior_summary.get)
    primary_behavior_name = config.behavior_classes.get(primary_behavior, "알 수 없음")
    print(f"\n주요 행동: {primary_behavior_name} ({behavior_summary[primary_behavior]} 프레임)")

    # 행동 분포 시각화
    visualize_behavior_distribution(behavior_summary, config)


def detect_behavior_transitions(predictions, frame_rate=30):
    """
    행동 전환을 탐지하고 전환 정보를 출력합니다.
    """
    transitions = []
    current_behavior = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_behavior:
            duration = (i - start_frame) / frame_rate
            transitions.append({
                "behavior": current_behavior,
                "start_frame": start_frame,
                "end_frame": i - 1,
                "duration": duration
            })
            current_behavior = predictions[i]
            start_frame = i

    # 마지막 행동 추가
    duration = (len(predictions) - start_frame) / frame_rate
    transitions.append({
        "behavior": current_behavior,
        "start_frame": start_frame,
        "end_frame": len(predictions) - 1,
        "duration": duration
    })

    print("\n행동 전환 분석:")
    for transition in transitions:
        behavior_name = config.behavior_classes.get(transition["behavior"], "알 수 없음")
        print(f"- 행동: {behavior_name}, 시작 프레임: {transition['start_frame']}, "
              f"종료 프레임: {transition['end_frame']}, 지속 시간: {transition['duration']:.2f}초")

    return transitions


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

def visualize_behavior_timeline(transitions, config, total_frames):
    """
    행동 전환 정보를 바탕으로 타임라인을 시각화합니다.
    """
    plt.figure(figsize=(12, 2))
    colors = plt.cm.Paired.colors

    for transition in transitions:
        start = transition["start_frame"]
        end = transition["end_frame"]
        behavior_id = transition["behavior"]
        behavior_name = config.behavior_classes.get(behavior_id, "알 수 없음")
        plt.plot([start, end], [1, 1], label=behavior_name, color=colors[behavior_id % len(colors)], linewidth=10)

    plt.title("행동 타임라인")
    plt.xlabel("프레임")
    plt.yticks([])  # Y축 숨기기
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def load_trained_model(config):
    checkpoint_path = "models/stgcn_behavior.pth"

    # 모델 초기화
    model = STGCN(
        in_channels=2,
        num_joints=config.num_joints,
        num_classes=config.num_classes,
        num_frames=30  # 고정된 프레임 수
    ).to(config.device)

    # 체크포인트 로드
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        print("체크포인트 로드 완료.")

        # 기존 가중치를 로드하면서 FC 레이어 제외
        pretrained_dict = checkpoint
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in model_dict and k != "fc.weight"
        }

        # 업데이트된 가중치 반영
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # FC 레이어 가중치 재초기화 (Xavier 방식)
        nn.init.xavier_uniform_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
        print("FC 레이어 가중치를 초기화했습니다.")
    else:
        print("체크포인트 파일이 없습니다. 새 모델을 사용합니다.")

    # 인접 행렬 생성 (단위 행렬 사용)
    adjacency_matrix = torch.eye(config.num_joints).to(config.device)

    return model, adjacency_matrix


def sample_frames(data, target_frames=30):
    """
    데이터의 프레임 수를 target_frames로 고정합니다.

    Args:
        data (np.ndarray): (frames, joints, 2) 형태의 데이터.
        target_frames (int): 고정할 프레임 수.

    Returns:
        np.ndarray: 고정된 프레임 수 데이터.
    """
    total_frames = data.shape[0]
    if total_frames < target_frames:
        # 부족한 경우 패딩
        padding = target_frames - total_frames
        data = np.pad(data, ((0, padding), (0, 0), (0, 0)), mode='constant')
    elif total_frames > target_frames:
        # 초과된 경우 샘플링
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        data = data[indices]
    return data


# ----- 모델 테스트 ----- 
def test_model_with_video(video_path):
    config = Config()

    # 스켈레톤 데이터 추출
    skeleton_data = extract_skeleton_from_video(video_path, config)
    if len(skeleton_data) == 0:
        print("영상에 관절 데이터가 감지되지 않았습니다. 다른 영상을 시도하세요.")
        return

    # 프레임 수 고정
    skeleton_data = sample_frames(skeleton_data, target_frames=30)

    # 데이터 변환
    skeleton_tensor = torch.tensor(skeleton_data, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(config.device)


    # 모델 로드 및 예측
    try:
        model, adjacency_matrix = load_trained_model(config)
        outputs = model(skeleton_tensor, adjacency_matrix)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        print(f"모든 프레임의 예측행동: {predictions}")

        # 행동 해석
        interpret_behavior(predictions, config)
        transitions = detect_behavior_transitions(predictions)
        visualize_behavior_timeline(transitions, config, total_frames=len(predictions))

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

    # 모델 및 인접 행렬 로드
    model, adjacency_matrix = load_trained_model(config)

    # 테스트를 진행합니다.
    try:
        test_model_with_video(config.video_path)
    except Exception as e:
        print(f"Error during model testing: {e}")
