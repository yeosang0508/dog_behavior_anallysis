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
TRAIN_CSV = r"data\csv_file\train_numeric.csv"
VAL_CSV = r"data\csv_file\train_numeric.csv"
TEST_CSV = r"data\csv_file\train_numeric.csv"

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
                 num_frames: int = 30,
                 hidden_size = 64,
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
        self.num_frames = num_frames
        self.hidden_size = hidden_size
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

        self.video_path = "video_test\8.mp4" # 테스트할 영상 파일 기본 경로    

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
            0: '몸 낮추기 - 방어적인 자세를 취하는 것처럼 보입니다.',
            1: '몸 긁기 - 반려견이 가려운 곳을 긁고 있어요.',
            2: '몸 흔들기 - 몸에 물기가 있을 때 흔드는 동작입니다.',
            3: '앞발 들기 - 주의 깊게 무언가를 관찰하고 있어요.',
            4: '한쪽 발 들기 - 불편하거나 흥미를 느끼는 동작일 수 있습니다.',
            5: '고개 돌리기 - 주변 상황을 살피고 있는 동작입니다.',
            6: '누워 있기 - 편안하거나 휴식을 취하는 모습이에요.',
            7: '마운팅 - 흥분 상태나 장난치는 행동일 수 있습니다.',
            8: '앉아 있기 - 반려견이 안정된 상태로 보입니다.',
            9: '꼬리 흔들기 - 기분이 좋거나 사람에게 관심을 보이는 모습이에요.',
            10: '돌아보기 - 주변 환경이나 사람을 확인하는 동작입니다.',
            11: '걷거나 뛰기 - 반려견이 활기차게 움직이고 있어요.'
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
    
    def get_behavior_description(self, behavior_class):
        """행동 클래스에 대한 설명을 반환합니다."""
        return self.behavior_classes.get(behavior_class, "알 수 없는 행동")
# 인스턴스 생성
config = Config()
