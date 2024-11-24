import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import sys

# 현재 파일의 경로를 기준으로 프로젝트의 루트 디렉토리를 가져옴
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from config.config import Config

# 통합 CSV 파일에서 데이터 시각화
def visualize_keypoints_from_csv(csv_path, output_folder, config):
    # CSV 파일 읽기
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # 각 프레임에 대해 시각화 수행
    for index, row in df.iterrows():
        frame_path = row["frame_path"]
        if not os.path.exists(frame_path):
            print(f"이미지 파일이 존재하지 않습니다: {frame_path}")
            continue

        # 키포인트 데이터 로드 (x1, y1, x2, y2, ... 형식이므로 이를 (x, y) 쌍으로 재구성)
        keypoints = []
        for i in range(2, len(row) - 1, 2):  # 2부터 시작하여 마지막 레이블 전까지 (2는 frame_path 이후 첫 키포인트 x좌표)
            x = row[i]
            y = row[i + 1]
            keypoints.append((x, y))

        # 이미지 로드
        image = cv2.imread(frame_path)
        if image is None:
            print(f"이미지 파일을 로드할 수 없습니다: {frame_path}")
            continue

        # 키포인트 및 관절 연결 시각화
        for i, (x, y) in enumerate(keypoints):
            if x is not None and y is not None:  # 키포인트가 존재할 때만 그리기
                color = config.joint_colors.get(i, (0, 255, 0))
                # 키포인트 그리기
                cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), radius=5, color=color, thickness=-1)
                # 키포인트 이름 표시
                label = config.joints_name.get(i, f"keypoint_{i}")
                cv2.putText(image, label, (int(x * image.shape[1]) + 5, int(y * image.shape[0]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 관절 연결 그리기
        for (start, end) in config.joint_pair:
            if keypoints[start] is not None and keypoints[end] is not None:
                x1, y1 = keypoints[start]
                x2, y2 = keypoints[end]
                color = config.joint_colors.get(start, (0, 255, 0))
                cv2.line(image, (int(x1 * image.shape[1]), int(y1 * image.shape[0])), (int(x2 * image.shape[1]), int(y2 * image.shape[0])), color=color, thickness=2)

        # 이미지 저장
        output_image_path = os.path.join(output_folder, f"frame_{index}.jpg")
        cv2.imwrite(output_image_path, image)
        print(f"이미지 저장 완료: {output_image_path}")

# 실행 코드 예시
if __name__ == "__main__":
    from config.config import Config

    # 설정 파일 인스턴스 생성
    config = Config()

    # CSV 파일 경로
    csv_path = "data/csv_file/annotations_combined.csv"
    output_folder = "outputs/visualizations/"

    # CSV 파일에서 키포인트 시각화
    visualize_keypoints_from_csv(csv_path, output_folder, config)
