# Dog Behavior Analysis 환경 설정 및 실행 가이드

## dog_classification 모델

### 1. 경로 설정

터미널에서 프로젝트 디렉터리로 이동합니다.

```bash
cd C:\Users\admin\IdeaProjects\TailsRoute_PJ\VScode
```

### 2. 가상 환경 생성 및 활성화

Python 가상 환경을 생성한 후 활성화합니다.

```bash
python -m venv dog_behavior_env
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser  # Windows에서 필요할 경우
.\dog_behavior_env\Scripts\Activate  # 가상 환경 활성화
```

### 3. 필수 패키지 설치

가상 환경 안에서 필요한 라이브러리들을 설치합니다.

```bash
# 프로젝트 루트 디렉터리에서 모델 디렉터리로 이동
cd dog_behavior_analysis/scripts/dog_classification/

# 해당 모델에 필요한 패키지 설치
pip install -r requirements.txt

# 원래 경로로 돌아가기
cd C:\Users\admin\IdeaProjects\TailsRoute_PJ\VScode
```

### 4. 모델 훈련을 위한 이미지 다운로드 및 스크립트 실행

preprocess.py 스크립트를 실행하여 이미지 데이터셋(zip 파일)을 다운로드합니다.

```bash
python dog_behavior_analysis/scripts/dog_classification/preprocess.py
```

### 5. 모델 훈련

데이터 전처리가 완료되면 train_model.py 스크립트를 실행하여 모델을 훈련시킵니다.

```bash
python dog_behavior_analysis/scripts/dog_classification/train_model.py
```

## 주의 사항

1. 가상 환경 활성화: 매번 프로젝트를 시작하기 전에 가상 환경을 활성화해야 합니다.

```bash
.\dog_behavior_env\Scripts\Activate
```

## behavior_analysis 모델
