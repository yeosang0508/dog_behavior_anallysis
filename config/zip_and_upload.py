import os
import shutil
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials

# 디렉토리를 ZIP 파일로 압축
def zip_directory(directory_path, output_zip_path):
    shutil.make_archive(output_zip_path.replace(".zip", ""), 'zip', directory_path)
    print(f"Directory {directory_path} compressed to {output_zip_path}")

# Google Drive에 업로드
def upload_to_my_drive(file_path):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    credentials = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    service = build('drive', 'v3', credentials=credentials)

    file_metadata = {'name': os.path.basename(file_path)}  # Google Drive에 저장될 파일 이름
    media = MediaFileUpload(file_path, resumable=True)

    # 파일 업로드
    uploaded_file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"File uploaded successfully. File ID: {uploaded_file.get('id')}")

# 실행
if __name__ == "__main__":
    directory_path = r"C:\Users\admin\IdeaProjects\test\VSCode\data"  # 업로드할 디렉토리 경로
    output_zip_path = r"C:\Users\admin\IdeaProjects\test\VSCode\data.zip"  # 압축 결과 경로

    # 1. 디렉토리를 ZIP 파일로 압축
    if os.path.exists(directory_path):
        zip_directory(directory_path, output_zip_path)

        # 2. Google Drive에 ZIP 파일 업로드
        upload_to_my_drive(output_zip_path)
    else:
        print(f"Error: Directory {directory_path} does not exist.")
