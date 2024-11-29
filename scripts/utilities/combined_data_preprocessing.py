import pandas as pd
import os
from sklearn.model_selection import train_test_split

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
def split_dataset(data, output_folder):
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
    validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])

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
    output_split_folder = "data/"

    # CSV 파일 통합 및 데이터셋 분리
    combined_data = merge_csv_files(csv_folder, output_combined_file)
    split_dataset(combined_data, output_split_folder)
