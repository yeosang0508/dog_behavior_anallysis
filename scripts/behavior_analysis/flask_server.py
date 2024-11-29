import sys
import os

# 프로젝트 최상위 디렉토리 추가
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(BASE_DIR)

from config.config import Config
from flask import Flask, request, jsonify
from video_behavior_analysis import test_model_with_video, calculate_behavior_duration, get_primary_behavior

config = Config()
app = Flask(__name__)

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
        
        # 행동 분석
        predictions = test_model_with_video(video_bytes)  # 메모리에서 직접 처리하는 방식으로 함수 수정 필요
        behavior_summary = calculate_behavior_duration(predictions)
        primary_behavior = get_primary_behavior(behavior_summary)

        print(f"[INFO] 분석 완료. 주요 행동: {primary_behavior}")
        result_message = "분석 완료: 행동 분석 결과입니다."
    except Exception as e:
        result_message = f"분석 중 오류 발생: {str(e)}"
        print(f"[ERROR] {result_message}")
        behavior_summary = None
        primary_behavior = None

    return jsonify({
        "message": result_message,
        "behavior_summary": behavior_summary,
        "primary_behavior": primary_behavior
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
