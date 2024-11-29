import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import requests
import numpy as np
import cv2
import imageio
from io import BytesIO
from flask import Flask, request, jsonify
from video_behavior_analysis import test_model_with_video, calculate_behavior_duration, get_primary_behavior
from config.config import Config
import tempfile


config = Config()

# Spring Boot API URL (결과를 보낼 URL)
SPRING_BOOT_API_URL = "http://localhost:8081/api/receive-analysis"

# Flask 애플리케이션 초기화
app = Flask(__name__)

# Video 파일을 메모리에서 읽어오는 함수 (cv2 사용)
def read_video_from_bytes(video_bytes):
    # BytesIO로 메모리 내 비디오 스트림 생성
    video_stream = BytesIO(video_bytes)
    
    # 임시 파일 경로 생성
    temp_file_path = tempfile.mktemp(suffix='.mp4')
    
    # 파일을 임시 경로에 저장
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(video_stream.getvalue())
    
    # 절대 경로로 변환하여 경로 인식 문제 해결
    temp_file_path = os.path.abspath(temp_file_path)
    
    # 임시 파일 경로가 제대로 생성되었는지 확인
    print(f"[INFO] 임시 파일 경로: {temp_file_path}")
    
    # cv2로 비디오 캡처
    cap = cv2.VideoCapture(temp_file_path)
    
    if not cap.isOpened():
        print(f"[ERROR] 비디오 파일 열기 실패: {temp_file_path}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # 프레임을 리스트에 추가
    
    cap.release()  # 캡처 객체 해제
    
    # 파일 사용 후 삭제
    os.remove(temp_file_path)
    
    return frames

# Flask의 '/analyze' 엔드포인트
@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        print(f"[INFO] 분석 시작. 모델 경로: models/stgcn_behavior.pth")

        # 파일을 메모리에서 바로 읽어서 분석
        video_bytes = file.read()
        
        # 메모리에서 비디오 읽기
        frames = read_video_from_bytes(video_bytes)
        
        if not frames:
            return jsonify({"error": "비디오 파일을 읽을 수 없습니다."}), 400
        
        # 행동 분석을 위한 예측 (프레임 데이터를 모델에 전달)
        predictions = test_model_with_video(frames)  # 수정된 함수는 프레임 리스트를 받음
        
        if predictions is None:
            return jsonify({"error": "모델 예측 실패"}), 500
        
        # 행동 지속 시간 계산
        behavior_summary = calculate_behavior_duration(predictions)
        
        if behavior_summary is None:
            return jsonify({"error": "행동 지속 시간 계산 실패"}), 500
        
        # 주요 행동 추출
        primary_behavior = get_primary_behavior(behavior_summary)
        
        if primary_behavior is None:
            return jsonify({"error": "주요 행동 추출 실패"}), 500

        print(f"[INFO] 분석 완료. 주요 행동: {primary_behavior}")
        result_message = "분석 완료: 행동 분석 결과입니다."

        # Spring Boot로 분석 결과 전송
        data_to_send = {
            "behavior_summary": behavior_summary,
            "primary_behavior": primary_behavior,
            "message": result_message
        }
        
        # Flask에서 Spring Boot로 분석 결과 전송
        response = requests.post(SPRING_BOOT_API_URL, json=data_to_send)

        if response.status_code == 200:
            print("Spring Boot로 결과 전송 완료.")
        else:
            print(f"Spring Boot로 결과 전송 실패. 상태 코드: {response.status_code}")

    except Exception as e:
        result_message = f"분석 중 오류 발생: {str(e)}"
        print(f"[ERROR] {result_message}")
        behavior_summary = None
        primary_behavior = None

    return jsonify({
        "message": result_message,
        "behavior_summary": behavior_summary or {},
        "primary_behavior": primary_behavior or {}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)