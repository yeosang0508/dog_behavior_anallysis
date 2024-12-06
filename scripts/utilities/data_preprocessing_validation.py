import os
import cv2
import pandas as pd


def verify_keypoint_mapping(csv_file, frame_dir, output_dir, crop_margin=0):
    """
    CSV 파일의 크롭 후 키포인트 매핑 정확도를 검증합니다.

    Parameters:
    - csv_file (str): CSV 파일 경로
    - frame_dir (str): 원본 프레임 이미지 디렉토리
    - output_dir (str): 시각화 결과 저장 디렉토리
    - crop_margin (int): 크롭 시 추가된 여백 (없으면 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    # CSV 로드
    data = pd.read_csv(csv_file)

    all_errors = []

    for index, row in data.iterrows():
        # 원본 프레임 경로 가져오기
        original_frame_path = os.path.join(frame_dir, row['cropped_frame_path'].lstrip("/\\"))
        if not os.path.exists(original_frame_path):
            print(f"Frame not found: {original_frame_path}")
            continue

        # 이미지 로드
        image = cv2.imread(original_frame_path)
        if image is None:
            print(f"Failed to load image: {original_frame_path}")
            continue

        # 크롭된 이미지의 가정된 오프셋 (필요에 따라 수정 가능)
        x_offset, y_offset = crop_margin, crop_margin
        h, w, _ = image.shape

        # 키포인트 좌표 변환 및 시각화
        original_keypoints = []
        transformed_keypoints = []
        for i in range(1, 16):
            x = row.get(f"x{i}")
            y = row.get(f"y{i}")
            if pd.notnull(x) and pd.notnull(y):
                x, y = int(float(x)), int(float(y))  # 원본 좌표
                transformed_x, transformed_y = x - x_offset, y - y_offset
                original_keypoints.append((x, y))
                transformed_keypoints.append((transformed_x, transformed_y))
                cv2.circle(image, (transformed_x, transformed_y), 5, (0, 255, 0), -1)  # 초록색 점
            else:
                original_keypoints.append(None)
                transformed_keypoints.append(None)

        # 오차 계산
        errors = calculate_keypoint_error(original_keypoints, transformed_keypoints)
        all_errors.append(errors)

        # 결과 저장
        output_path = os.path.join(output_dir, f"verified_{row['frame_number']}.jpg")
        if not cv2.imwrite(output_path, image):
            print(f"Failed to save visualization: {output_path}")
        else:
            print(f"Saved visualization: {output_path}")

    # 전체 평균 오차 계산
    mean_error = calculate_mean_error(all_errors)
    print(f"Overall Mean Error: {mean_error:.2f}")


def calculate_keypoint_error(original_keypoints, transformed_keypoints):
    """
    키포인트의 크롭 전후 오차를 계산합니다.

    Parameters:
    - original_keypoints (list of tuple): 원본 키포인트 좌표
    - transformed_keypoints (list of tuple): 변환된 키포인트 좌표

    Returns:
    - errors (list): 각 키포인트의 유클리디안 거리
    """
    errors = []
    for orig, trans in zip(original_keypoints, transformed_keypoints):
        if orig is not None and trans is not None:
            error = ((orig[0] - trans[0]) ** 2 + (orig[1] - trans[1]) ** 2) ** 0.5
            errors.append(error)
        else:
            errors.append(None)  # 키포인트가 없을 경우 None
    return errors


def calculate_mean_error(all_errors):
    """
    전체 키포인트의 평균 오차를 계산합니다.

    Parameters:
    - all_errors (list of list): 각 이미지의 키포인트 오차 리스트

    Returns:
    - mean_error (float): 전체 키포인트 평균 오차
    """
    flat_errors = [e for errors in all_errors for e in errors if e is not None]
    return sum(flat_errors) / len(flat_errors) if flat_errors else 0.0


if __name__ == "__main__":
    # CSV 파일 경로 및 데이터 경로 설정
    csv_file = "data/csv_file/annotations_sit.csv"  # CSV 파일 경로
    frame_dir = "data/train/mounting/frame_mounting"  # 원본 프레임 디렉토리
    output_dir = "data/keypoint_verifications"  # 검증 결과 저장 디렉토리

    # 크롭 여백 설정 (필요하면 수정 가능)
    crop_margin = 10

    # 검증 함수 실행
    verify_keypoint_mapping(csv_file, frame_dir, output_dir, crop_margin)
