# CSV 읽기
import os
import pandas as pd

df = pd.read_csv(r"data\csv_file\train_numeric.csv")
file_path = r"data\csv_file\train_numeric.csv"


# 누락된 값 확인
print(df.isnull().sum())
print(f"데이터 크기: {df.shape}")
print(f"파일 크기: {os.path.getsize(file_path)} bytes")