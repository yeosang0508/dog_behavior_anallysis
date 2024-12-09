import cv2

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: 비디오 파일을 열 수 없습니다.")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    
    if len(frames) == 0:
        print("Error: 프레임을 하나도 로드하지 못했습니다.")
    else:
        print(f"총 {len(frames)}개의 프레임이 로드되었습니다.")
    
    return frames

# 사용 예시
video_path = "C:/Users/admin/IdeaProjects/test/VSCode/data/video1/12.mp4"
loaded_frames = load_video_frames(video_path)

# 비디오 프레임 확인
if loaded_frames:
    height, width = loaded_frames[0].shape[:2]
    print(f"프레임 크기: {width}x{height}")
else:
    print("프레임 로드에 실패했습니다.")
