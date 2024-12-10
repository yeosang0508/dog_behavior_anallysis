import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

# 행동 라벨 정의 (13개)
behavior_classes = {
    "bodylower": 0,
    "bodyscratch": 1,
    "bodyshake": 2,
    "feetup": 3,
    "footup": 4,
    "heading": 5,
    "lying": 6,
    "mounting": 7,
    "sit": 8,
    "tailing": 9,
    "taillow": 10,
    "turn": 11,
    "walkrun": 12
}

# CSV 파일 통합 및 저장
def merge_csv_files(csv_folder, output_file):
    """
    여러 행동 폴더에 있는 CSV 파일을 하나로 통합합니다.
    """
    csv_files = {
        "bodylower": "bodylower.csv",
        "bodyscratch": "bodyscratch.csv",
        "bodyshake": "bodyshake.csv",
        "feetup": "feetup.csv",
        "footup": "footup.csv",
        "heading": "heading.csv",
        "lying": "lying.csv",
        "mounting": "mounting.csv",
        "sit": "sit.csv",
        "tailing": "tailing.csv",
        "taillow": "taillow.csv",
        "turn": "turn.csv",
        "walkrun": "walkrun.csv"
    }

    all_data = []
    for label, filename in csv_files.items():
        file_path = os.path.join(csv_folder, filename)
        if not os.path.exists(file_path):
            print(f"파일이 존재하지 않습니다: {file_path}")
            continue

        data = pd.read_csv(file_path)
        all_data.append(data)

    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.to_csv(output_file, index=False)
    print(f"통합 데이터가 저장되었습니다: {output_file}")
    return combined_data


# 실행
if __name__ == "__main__":
    # 경로 설정
    csv_folder = "data/csv_file/"  # 행동 폴더별 CSV가 저장된 디렉토리
    output_combined_file = "data/csv_file/combined.csv"  # 통합 CSV 경로

    # CSV 파일 통합
    print("CSV 파일 통합 중...")
    combined_data = merge_csv_files(csv_folder, output_combined_file)

