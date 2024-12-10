import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 행동 라벨 정의 (정수형으로 변환)
behavior_classes = {
    "bodyshake": 0,
    "heading": 1,
    "lying": 2,
    "mounting": 3,
    "sit": 4
}

# CSV 파일 통합 및 저장
def merge_csv_files(csv_folder, output_file):
    """
    여러 행동 폴더에 있는 CSV 파일을 하나로 통합합니다.
    """
    csv_files = {
        "bodyshake": "annotations_bodyshake.csv",
        "heading": "annotations_heading.csv",
        "lying": "annotations_lying.csv",
        "mounting": "annotations_mounting.csv",
        "sit": "annotations_sit.csv"
    }

    all_data = []
    for label, filename in csv_files.items():
        file_path = os.path.join(csv_folder, filename)
        if not os.path.exists(file_path):
            print(f"파일이 존재하지 않습니다: {file_path}")
            continue

        data = pd.read_csv(file_path)
        data['behavior_label'] = label
        data['behavior_class'] = behavior_classes[label]
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
    print(f"통합 데이터가 저장되었습니다: {output_file}")
    return combined_data

# 데이터셋 분리
def split_dataset(data, output_folder):
    """
    데이터셋을 훈련, 검증, 테스트 세트로 분리합니다.
    """
    # 데이터셋 분리
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['behavior_class'])
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['behavior_class'])

    print(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}, Test size: {len(test_data)}")
    print(f"클래스 비율 (훈련): \n{train_data['behavior_class'].value_counts(normalize=True)}")

    # 데이터 저장
    os.makedirs(output_folder, exist_ok=True)
    train_data.to_csv(os.path.join(output_folder, "annotations_train.csv"), index=False)
    validation_data.to_csv(os.path.join(output_folder, "annotations_validation.csv"), index=False)
    test_data.to_csv(os.path.join(output_folder, "annotations_test.csv"), index=False)

    print("훈련, 검증, 테스트 데이터가 저장되었습니다!")
    return train_data, validation_data, test_data

# 실행 코드
if __name__ == "__main__":
    # 경로 설정
    csv_folder = "data/csv_file/"  # 행동 폴더별 CSV가 저장된 디렉토리
    output_combined_file = "data/csv_file/annotations_combined.csv"  # 통합 CSV 경로
    output_split_folder = "data/split_data/"  # 분리된 데이터 저장 폴더

    # CSV 파일 통합
    print("CSV 파일 통합 중...")
    combined_data = merge_csv_files(csv_folder, output_combined_file)

    # 라벨 값 검증
    print(f"라벨 값의 범위: {combined_data['behavior_class'].min()} ~ {combined_data['behavior_class'].max()}")

    # 데이터셋 분리
    print("데이터셋 분리 중...")
    train_data, validation_data, test_data = split_dataset(combined_data, output_split_folder)
