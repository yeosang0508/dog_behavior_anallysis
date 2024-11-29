import os
import numpy as np
import torch
from typing import List
import pandas as pd 
import matplotlib.pyplot as plt


# 프로젝트 최상위 디렉토리
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 경로 설정
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_CSV = os.path.join(DATA_DIR, 'annotations_train.csv')
VAL_CSV = os.path.join(DATA_DIR, 'annotations_validation.csv')
TEST_CSV = os.path.join(DATA_DIR, 'annotations_test.csv')

# 모델 경로 설정
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BEHAVIOR_MODELS_DIR = os.path.join(MODELS_DIR, 'dog_behavior')

# 출력 경로 설정
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')
VISUALIZATIONS_DIR = os.path.join(OUTPUTS_DIR, 'visualizations')

# 필요한 디렉토리 생성
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

class Config:
    def __init__(self,
                 input_size: List[int] = [384, 288],
                 kpd: float = 4.0,
                 epochs: int = 15,
                 sigma: float = 3.0,
                 num_joints: int = 15,
                 batch_size: int = 16,
                 random_seed: int = 2021,
                 test_ratio: float = 0.1,
                 learning_rate: float = 1e-3,
                 save_folder: str = '',
                 loss_type: str = "MSE",
                 target_type: str = "gaussian",
                 post_processing: str = "dark",
                 debug: bool = False,
                 shift: bool = False,
                 init_training: bool = False,
                 train_csv=TRAIN_CSV,
                 val_csv=VAL_CSV,
                 test_csv=TEST_CSV):



        # CSV 파일 경로를 속성으로 설정
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv

        # num_classes 자동 계산
        self.num_classes = self._calculate_num_classes()

        # 기타 설정
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder) and self.save_folder != '':
            os.makedirs(self.save_folder, exist_ok=True)

        self.num_epochs = epochs
        self.seed = random_seed
        self.learning_rate = learning_rate
        self.loss_type = loss_type
        self.num_joints = num_joints
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.init_training = init_training
        self.kpd = kpd
        self.sigma = sigma
        self.shift = shift
        self.debug = debug
        self.target_type = target_type
        self.image_size = np.array(input_size)
        self.output_size = self.image_size // 4
        self.post_processing = post_processing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.video_path = "video_test\stand.mp4" # 테스트할 영상 파일 기본 경로    

        # 모델 저장 디렉토리
        self.models_dir = os.path.join(os.getcwd(), 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # CSV 열 이름 정의 (x, y 좌표)
        self.joint_columns = [f'x{i}' for i in range(1, self.num_joints + 1)] + \
                             [f'y{i}' for i in range(1, self.num_joints + 1)]

        # 관절 이름 설정
        self.joints_name = {
            0: '코', 1: '이마 중앙', 2: '입꼬리', 3: '아래 입술 중앙', 4: '목',
            5: '오른쪽 앞다리 시작', 6: '왼쪽 앞다리 시작', 7: '오른쪽 앞다리 발목', 8: '왼쪽 앞다리 발목',
            9: '오른쪽 대퇴부', 10: '왼쪽 대퇴부', 11: '오른쪽 뒷다리 발목', 12: '왼쪽 뒷다리 발목',
            13: '꼬리 시작', 14: '꼬리 끝'
        }

        self.behavior_classes = {
            0: '몸 낮추기',         # bodylower
            1: '몸 긁기',           # bodyscratch
            2: '몸 흔들기',         # bodyshake
            3: '앞발 들기',         # feetup
            4: '한쪽 발 들기',      # footup
            5: '고개 돌리기',       # heading
            6: '누워 있기',         # lying
            7: '마운팅',            # mounting
            8: '앉아 있기',         # sit
            9: '꼬리 흔들기',       # tailing
            10: '돌아보기',         # turn
            11: '걷거나 뛰기'       # walkrun
        }

        # 관절 연결 정보 (pair-wise connection)
        self.joint_pair = [
            (0, 1),  (0, 3), (2, 3), (3, 4), (4, 5),
            (4, 6), (5, 7), (6, 8), (4, 13), (13, 9),
            (13, 10), (13, 14), (9, 11), (10, 12)
        ]

        # 좌우 대칭 관절 쌍
        self.flip_pair = [
            (9, 10), (7, 8), (5, 6), (11, 12)
        ]

        # 관절별 가중치
        self.joints_weight = np.array(
            [
                1.3,  # 코
                1.3,  # 이마 중앙
                1.3,  # 입꼬리(입끝)
                1.3,  # 아래 입술 중앙
                1.3,  # 목
                1.3, 1.3,  # 앞다리 시작
                1.3, 1.3,  # 앞다리 발목
                1.3, 1.3,  # 대퇴골
                1.3, 1.3,  # 뒷다리 발목
                1.3,  # 꼬리 시작
                1.3  # 꼬리 끝
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        # 시각화 색상 정의
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, self.num_joints + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        self.joint_colors = {k: colors[k] for k in range(self.num_joints)}

    def _calculate_num_classes(self):
        """CSV 파일에서 고유한 레이블의 개수를 계산."""
        data = pd.read_csv(self.train_csv)
        unique_classes = data['label'].nunique()
        print(f"Number of unique classes (num_classes): {unique_classes}")
        return unique_classes
    
# 인스턴스 생성
config = Config()
