import pandas as pd

# 행동 클래스 매핑
behavior_classes = {
    0: 'bodylower',  # bodylower
    1: 'bodyscratch',  # bodyscratch
    2: 'bodyshake',  # bodyshake
    3: 'feetup',  # feetup
    4: 'footup',  # footup
    5: 'heading',  # heading
    6: 'lying',  # lying
    7: 'mounting',  # mounting
    8: 'sit',  # sit
    9: 'tailing',  # tailing
    10: 'turn',  # turn
    11: 'walkrun'  # walkrun
}

# 텍스트 -> 숫자 매핑
label_mapping = {v: k for k, v in behavior_classes.items()}

# CSV 파일 로드
csv_file = r"data\csv_file\annotations_combined.csv"
output_file = r"data\csv_file\train_numeric.csv"

# CSV 파일 읽기 (헤더 유지)
data = pd.read_csv(csv_file, header=0)

# 텍스트 레이블을 숫자로 변환 (마지막 열)
data.iloc[:, -1] = data.iloc[:, -1].map(label_mapping)

# 정수형 변환
data.iloc[:, -1] = data.iloc[:, -1].fillna(-1).astype(int)

# 변환된 데이터 확인
print(data.head())

# 데이터프레임을 CSV로 저장 (헤더 포함)
data.to_csv(output_file, index=False)
print(f"새로운 csv 파일 {output_file}이 저장되었습니다.")
