import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import cv2
import torch
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip
from stgcn_model import UnifiedModel3DCNN
from config.config import Config
from PIL import ImageFont, ImageDraw, Image
# Config 인스턴스 생성
config = Config()


def reencode_video(input_path, output_path="processed_video.mp4"):
    """
    입력 비디오를 재인코딩하여 OpenCV와 호환 가능한 비디오를 생성합니다.
    """
    try:
        print(f"[INFO] 비디오 재인코딩 중: {input_path} -> {output_path}")
        clip = VideoFileClip(input_path)
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"[INFO] 재인코딩된 비디오가 생성되었습니다: {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] 비디오 재인코딩 실패: {e}")
        return None


def extract_skeleton_from_video(video_path, config, num_joints=None):
    """
    비디오에서 스켈레톤 데이터를 추출합니다.
    MediaPipe를 사용하여 관절 데이터를 추출합니다.
    """
    if num_joints is None:
        num_joints = config.num_joints

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 비디오 정보 - Total Frames: {total_frames}, Width: {width}, Height: {height}, FPS: {fps}")

    skeleton_data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        keypoints = []
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y) for lm in landmarks]
            print(f"[INFO] Frame {frame_idx}: {len(keypoints)} landmarks detected.")
        else:
            print(f"[WARNING] Frame {frame_idx}: No landmarks detected.")
            keypoints = [(0.0, 0.0)] * num_joints

        if len(keypoints) != num_joints:
            if len(keypoints) < num_joints:
                keypoints += [(0.0, 0.0)] * (num_joints - len(keypoints))
            else:
                keypoints = keypoints[:num_joints]

        skeleton_data.append(keypoints)
        frame_idx += 1

    cap.release()
    pose.close()
    print(f"[INFO] 총 {frame_idx} 프레임이 처리되었습니다.")
    return np.array(skeleton_data, dtype=np.float32)


def sample_frames(data, target_frames=None):
    """
    데이터를 특정 프레임 수로 고정하거나, 전체 데이터를 유지합니다.
    """
    if target_frames is None:
        # 모든 데이터를 그대로 반환
        return data

    total_frames = data.shape[0]
    if total_frames < target_frames:
        # 부족한 경우 패딩
        padding = target_frames - total_frames
        data = np.pad(data, ((0, padding), (0, 0), (0, 0)), mode='constant')
    elif total_frames > target_frames:
        # 초과한 경우 균등 샘플링
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        data = data[indices]
    return data


def put_korean_text(image, text, position, font_path="malgun.ttf", font_size=20, color=(255, 255, 255)):
    """
    OpenCV 이미지에 한글 텍스트를 추가합니다.
    """
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def test_samurai_model_with_video(video_path, config):
    """
    SAMURAI 모델로 행동을 예측합니다. 모든 프레임에 대해 예측을 수행합니다.
    """
    # 스켈레톤 데이터 추출
    skeleton_data = extract_skeleton_from_video(video_path, config)
    if skeleton_data is None or len(skeleton_data) == 0:
        print("[ERROR] 스켈레톤 데이터를 추출할 수 없습니다.")
        return None, None

    # 스켈레톤 데이터를 텐서로 변환
    skeleton_tensor = (
        torch.tensor(skeleton_data, dtype=torch.float32)
        .permute(2, 0, 1)  # (frames, joints, 2) -> (2, frames, joints)
        .unsqueeze(0)      # (2, frames, joints) -> (1, 2, frames, joints)
        .to(config.device)
    )

    # 모델 경로 설정
    model_path = os.path.join(config.models_dir, "stgcn_behavior.pth")
    print(f"[DEBUG] 모델 경로: {model_path}")

    # 모델 경로 확인
    if not os.path.exists(model_path):
        print(f"[ERROR] 모델 파일이 존재하지 않습니다: {model_path}")
        exit()

    # 모델 로드
    model = UnifiedModel3DCNN(
        in_channels=2,
        num_joints=config.num_joints,
        num_classes=config.num_classes,
        num_frames=config.num_frames,
        hidden_size=config.hidden_size
    ).to(config.device)

    checkpoint = torch.load(model_path, map_location=config.device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # 모델 예측
    predictions = []
    with torch.no_grad():
        adjacency_matrix = torch.eye(config.num_joints).to(config.device)
        outputs = model(skeleton_tensor, adjacency_matrix)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    return predictions, skeleton_data

def detect_and_track_objects(video_path, config, output_path="object_tracking.mp4"):
    """
    객체 탐지와 관절 추적을 결합하여 객체 내 관절을 추적합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")
        return

    # 객체 탐지 모델 로드 (YOLOv5 사용 예)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [16]  # 강아지(객체 ID 16)로 제한

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)  # YOLO로 객체 탐지
        detections = results.xyxy[0].cpu().numpy()  # 객체 경계 상자
        dog_boxes = [box for box in detections if box[-1] == 16]  # 강아지 객체 필터링

        if dog_boxes:
            for box in dog_boxes:
                x1, y1, x2, y2, conf, _ = box.astype(int)
                cropped_frame = frame[y1:y2, x1:x2]  # 강아지 영역 잘라내기
                frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(frame_rgb)

                if pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark
                    for idx, lm in enumerate(landmarks):
                        cx, cy = int(lm.x * cropped_frame.shape[1]), int(lm.y * cropped_frame.shape[0])
                        cv2.circle(frame, (x1 + cx, y1 + cy), radius=5, color=(0, 0, 255), thickness=-1)
                        joint_name = config.joints_name.get(idx, f"Joint {idx}")
                        cv2.putText(frame, joint_name, (x1 + cx + 5, y1 + cy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[INFO] 객체 탐지 및 관절 추적 결과가 '{output_path}'에 저장되었습니다.")


def analyze_behavior_frequency(predictions, config):
    """
    예측된 행동의 발생 빈도를 계산하고, 가장 많이 발생한 행동을 반환합니다.
    """
    from collections import Counter

    counter = Counter(predictions)
    most_common_behavior = counter.most_common(1)[0]  # 가장 많이 발생한 행동 (behavior_id, count)

    print("[INFO] 행동 빈도 분석:")
    for behavior_id, count in counter.items():
        print(f"- {config.behavior_classes.get(behavior_id, 'Unknown')}: {count}번 발생")

    return most_common_behavior


def visualize_predictions_with_frequent_behavior(
    video_path,
    predictions,
    skeleton_data,
    most_common_behavior,
    config,
    output_path="output_with_behavior_analysis.mp4",
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 비디오를 열 수 없습니다: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수
    frame_idx = 0

    # predictions를 total_frames 길이에 맞게 확장
    predictions = np.resize(predictions, total_frames)
    skeleton_data = np.resize(skeleton_data, (total_frames, *skeleton_data.shape[1:]))

    behavior_label_frequent = config.behavior_classes.get(
        most_common_behavior[0], "Unknown"
    )
    behavior_count_frequent = most_common_behavior[1]

    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        behavior_label = config.behavior_classes.get(
            predictions[frame_idx], "Unknown"
        )
        keypoints = skeleton_data[frame_idx]

        # 관절 간 선 그리기
        for start_idx, end_idx in config.joint_pair:
            x1, y1 = int(keypoints[start_idx][0] * frame.shape[1]), int(keypoints[start_idx][1] * frame.shape[0])
            x2, y2 = int(keypoints[end_idx][0] * frame.shape[1]), int(keypoints[end_idx][1] * frame.shape[0])
            if (x1 > 0 and y1 > 0) and (x2 > 0 and y2 > 0):
                cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # 관절 이름과 원 표시
        for idx, (x, y) in enumerate(keypoints):
            x, y = int(x * frame.shape[1]), int(y * frame.shape[0])
            if x > 0 and y > 0:
                joint_name = config.joints_name.get(idx, f"Joint {idx}")
                cv2.putText(frame, joint_name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

        # 좌우 대칭 관절 표시
        for left_idx, right_idx in config.flip_pair:
            xl, yl = int(keypoints[left_idx][0] * frame.shape[1]), int(keypoints[left_idx][1] * frame.shape[0])
            xr, yr = int(keypoints[right_idx][0] * frame.shape[1]), int(keypoints[right_idx][1] * frame.shape[0])
            if (xl > 0 and yl > 0):
                cv2.circle(frame, (xl, yl), radius=6, color=(0, 255, 255), thickness=2)
            if (xr > 0 and yr > 0):
                cv2.circle(frame, (xr, yr), radius=6, color=(255, 255, 0), thickness=2)

        # 현재 프레임의 행동 표시
        cv2.putText(
            frame,
            f"Behavior: {behavior_label}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        # 가장 많이 발생한 행동 표시
        cv2.putText(
            frame,
            f"Most Frequent Behavior: {behavior_label_frequent} ({behavior_count_frequent} times)",
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[INFO] 결과 비디오가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # 1. 입력 비디오 경로 설정
    video_path = config.video_path

    # 2. 객체 탐지 및 관절 추적 실행 (강아지 탐지 및 시각화)
    object_tracking_output = "object_tracking_output.mp4"
    detect_and_track_objects(video_path, config, output_path=object_tracking_output)

    # 3. SAMURAI 모델로 행동 예측 실행
    predictions, skeleton_data = test_samurai_model_with_video(video_path, config)

    # 4. 행동 예측 결과 확인 및 시각화
    if predictions is None:
        print("[ERROR] 행동 예측 실패.")
    else:
        most_common_behavior = analyze_behavior_frequency(predictions, config)
        print(
            f"[INFO] 가장 많이 발생한 행동: {config.behavior_classes.get(most_common_behavior[0], 'Unknown')} ({most_common_behavior[1]}번 발생)"
        )

        # 행동 예측 결과 시각화
        visualize_predictions_with_frequent_behavior(
            video_path, predictions, skeleton_data, most_common_behavior, config
        )

    # 5. 전체 프로세스 완료
    print("[INFO] 작업 완료.")
