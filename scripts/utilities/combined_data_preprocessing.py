import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import numpy as np

# 데이터 증강 함수
def augment_data(data, num_augmentations=1):
    """
    데이터 증강: 좌우 반전과 같은 간단한 변환 수행.
    Args:
        data (pd.DataFrame): 증강할 데이터
        num_augmentations (int): 증강 반복 횟수
    Returns:
        pd.DataFrame: 증강된 데이터
    """
    augmented_data = []
    for _ in range(num_augmentations):
        flipped = data.copy()
        for col in flipped.columns:
            if 'x' in col:  # x 좌표는 좌우 반전
                flipped[col] = -flipped[col]
        augmented_data.append(flipped)
    return pd.concat(augmented_data, ignore_index=True)

# CSV 파일 통합 및 저장
def merge_csv_files(csv_folder, output_file):
    csv_files = {
        "bodylower": "annotations_bodylower.csv",
        "bodyscratch": "annotations_bodyscratch.csv",
        "bodyshake": "annotations_bodyshake.csv",
        "feetup": "annotations_feetup.csv",
        "footup": "annotations_footup.csv",
        "heading": "annotations_heading.csv",
        "lying": "annotations_lying.csv",
        "mounting": "annotations_mounting.csv",
        "sit": "annotations_sit.csv",
        "tailing": "annotations_tailing.csv",
        "turn": "annotations_turn.csv",
        "walkrun": "annotations_walkrun.csv"
    }

    all_data = []
    for label, filename in csv_files.items():
        file_path = os.path.join(csv_folder, filename)
        data = pd.read_csv(file_path)
        data['label'] = label  # 레이블 추가
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")
    return combined_data

# 데이터셋 분리 및 저장
def split_dataset(data, output_folder, apply_augmentation=True, augment_factor=1):
    """
    데이터셋을 분리하고 데이터 증강 및 오버샘플링 적용.
    Args:
        data (pd.DataFrame): 원본 데이터셋
        output_folder (str): 저장할 폴더 경로
        apply_augmentation (bool): 데이터 증강 여부
        augment_factor (int): 증강 데이터 생성 반복 횟수
    Returns:
        tuple: 훈련, 검증, 테스트 데이터셋
    """
    # 데이터셋 분리
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['label'])
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])

    # 클래스 비율 확인
    print("Before augmentation:")
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(validation_data)}")
    print(f"Test size: {len(test_data)}")
    print(train_data['label'].value_counts(normalize=True))

    # 데이터 증강
    if apply_augmentation:
        print("Applying data augmentation...")
        augmented_data = augment_data(train_data, num_augmentations=augment_factor)
        train_data = pd.concat([train_data, augmented_data], ignore_index=True)
        print(f"After augmentation, train size: {len(train_data)}")
        print(train_data['label'].value_counts(normalize=True))

    # 오버샘플링
    print("Applying oversampling...")
    ros = RandomOverSampler(random_state=42)
    train_data, train_labels = ros.fit_resample(train_data, train_data['label'])
    train_data['label'] = train_labels  # 오버샘플링 후 라벨 복원
    print(f"After oversampling, train size: {len(train_data)}")
    print(train_data['label'].value_counts(normalize=True))

    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(train_data['label']), y=train_data['label'])
    print(f"Class Weights: {class_weights}")

    # 데이터 저장
    os.makedirs(output_folder, exist_ok=True)
    train_data.to_csv(os.path.join(output_folder, "annotations_train.csv"), index=False)
    validation_data.to_csv(os.path.join(output_folder, "annotations_validation.csv"), index=False)
    test_data.to_csv(os.path.join(output_folder, "annotations_test.csv"), index=False)

    print("저장 완료!")
    return train_data, validation_data, test_data

if __name__ == "__main__":
    # 경로 설정
    csv_folder = "data/csv_file/"
    output_combined_file = "data/csv_file/annotations_combined.csv"
    output_split_folder = "data/split_data"

    # CSV 파일 통합 및 데이터셋 분리
    combined_data = merge_csv_files(csv_folder, output_combined_file)
    split_dataset(combined_data, output_split_folder, apply_augmentation=True, augment_factor=2)
