from pathlib import Path
import pandas as pd
import torch
from torch import nn

# 시스템 설정 및 경로 정의
torch.set_num_threads(1)  # CPU 연산 시 일관된 테스트 환경을 위해 스레드 수를 1개로 고정
base = Path(__file__).resolve().parent  # 현재 파이썬 파일이 위치한 폴더의 절대 경로 확보

# 데이터 로드 및 분할 (Pandas)
df = pd.read_csv(base / "points.csv")  # points.csv 파일을 읽어와 데이터프레임 생성
train = df[df["split"] == "train"]  # 학습에 사용할 'train' 데이터 분리
valid = df[df["split"] == "valid"]  # 검증에 사용할 'valid' 데이터 분리

# 데이터프레임을 PyTorch 텐서(Tensor)로 변환
# 입력(x1, x2)은 실수형(float32), 정답(label)은 클래스 번호이므로 정수형(long)으로 설정합니다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# 학습률 후보 두 개를 리스트로 쓰세요: 0.001, 0.01.
# 모델·seed·20 epochs·AdamW·weight_decay=0은 그대로 둡니다.
# >>> B03
# TODO B03: 위 요구사항에 맞게 이 블록만 완성합니다.
learning_rates = [0.001, 0.01]  # 실험할 두 가지 학습률(보폭) 후보 정의
# <<< B03

# 손실 함수(Loss Function) 정의 (다중 분류용 교차 엔트로피)
loss_fn = nn.CrossEntropyLoss()

# 설정한 학습률 후보들을 하나씩 꺼내어 비교 실험 반복
for lr in learning_rates:
    torch.manual_seed(42)  # 각 학습률 실험마다 동일한 가중치에서 시작하도록 랜덤 시드 초기화
    
    # 신경망 구조 정의 (입력 2 -> 은닉 8 + ReLU -> 출력 2)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    
    # 꺼내온 학습률(lr)을 적용하여 AdamW 옵티마이저 생성
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # 매 에포크마다 계산되는 검증 손실(Valid Loss)을 차곡차곡 저장할 빈 리스트
    losses = []
    
    # 총 20회(Epoch) 학습 진행
    for epoch in range(20):
        # --- 1. 모델 학습 단계 ---
        model.train()         # 모델을 학습 모드로 전환
        optimizer.zero_grad()  # 이전 루프에서 계산된 가중치의 기울기(Gradient)를 0으로 초기화
        
        # 순전파(예측값 계산) -> 손실 계산 -> 역전파(기울기 계산)를 한 줄로 수행
        loss_fn(model(X_train), y_train).backward()
        
        optimizer.step()      # 계산된 기울기를 바탕으로 가중치 업데이트(갱신)
        
        # --- 2. 실시간 모델 평가 단계 (매 에포크 종료 직후) ---
        # 갱신이 끝난 모델의 valid loss를 기록합니다. valid로 backward하지 않습니다.
        model.eval()          # 모델을 평가 모드로 전환
        with torch.no_grad(): # 평가 시에는 미분(기울기 계산) 메모리를 쓰지 않도록 설정
            # 검증 데이터를 모델에 넣어 오차를 구하고, .item()으로 숫자만 추출
            valid_loss = loss_fn(model(X_valid), y_valid).item()
            losses.append(valid_loss)  # 기록용 리스트에 검증 손실 값 추가
            
    # 한 학습률에 대한 20회 반복이 끝나면 결과 출력
    # 처음 5개 에포크의 검증 손실을 소수점 6자리까지 반올림하여 확인
    print("lr:", lr, "처음 5개:", [round(x, 6) for x in losses[:5]])
    # 20번째(마지막) 에포크의 최종 검증 손실 확인 (losses[-1]은 리스트의 가장 마지막 원소)
    print("마지막 valid loss:", round(losses[-1], 6))

    """ 출력 결과 값
    lr: 0.001 처음 5개: [0.50841, 0.506854, 0.505305, 0.503762, 0.502226]
    마지막 valid loss: 0.480205
    lr: 0.01 처음 5개: [0.494631, 0.480118, 0.46654, 0.453585, 0.442029]
    마지막 valid loss: 0.307332 """