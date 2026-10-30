from pathlib import Path
import pandas as pd
import torch
from torch import nn

# [1] 시스템 설정 및 경로 정의
torch.set_num_threads(1)  # CPU 연산 시 사용할 스레드 수를 1개로 제한 (일관된 테스트 환경용)
base = Path(__file__).resolve().parent  # 현재 실행 중인 파이썬 파일의 폴더 경로를 절대경로로 획득

# [2] 데이터 로드 및 분할 (Pandas)
df = pd.read_csv(base / "points.csv")  # 같은 폴더에 있는 points.csv 파일을 읽어와 데이터프레임으로 변환
train = df[df["split"] == "train"]  # split 열의 값이 'train'인 행만 골라내어 학습 데이터로 지정
valid = df[df["split"] == "valid"]  # split 열의 값이 'valid'인 행만 골라내어 검증 데이터로 지정

# [3] Pandas 데이터프레임을 PyTorch 텐서(Tensor)로 변환
# .values로 numpy 배열을 만든 뒤 torch.tensor()로 변환합니다.
# x1, x2는 입력 특성이므로 실수형(float32), 정답 레이블(label)은 클래스 번호이므로 정수형(long)을 사용합니다.
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# [4] 손실 함수(Loss Function) 정의
# 다중 클래스 분류 문제이므로 교차 엔트로피 손실 함수를 사용합니다.
loss_fn = nn.CrossEntropyLoss()
for name in ["SGD", "AdamW"]:
    # 후보마다 같은 초기 가중치에서 다시 시작합니다. 앞 모델을 이어 학습하지 않습니다.
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    if name == "SGD":
        # SGD(Stochastic Gradient Descent, 확률적 경사하강법) 최적화기를 생성합니다.
        # loss.backward()가 각 파라미터의 .grad에 저장한 기울기를 이용하여,
        # optimizer.step()을 호출할 때 가중치와 편향을 갱신합니다.
        # 한 번에 사용할 데이터의 개수는 SGD가 아니라 DataLoader의 batch_size로 정합니다.
        optimizer = torch.optim.SGD(

            # 갱신할 모델의 파라미터를 전달합니다.
            # 현재 모델에서는 각 Linear 층의 가중치(weight)와 편향(bias)이 해당합니다.
            # ReLU처럼 학습할 파라미터가 없는 층은 갱신 대상이 없습니다.
            model.parameters(),

            # 학습률(learning rate): 기울기를 이용해 파라미터를 얼마나 이동시킬지 정합니다.
            # momentum과 weight_decay가 0이면 다음 식으로 갱신합니다.
            # 새로운 파라미터 = 기존 파라미터 - 학습률 × 현재 기울기
            # 너무 크면 손실이 진동하거나 발산할 수 있고, 너무 작으면 학습이 느릴 수 있습니다.
            lr=0.01,

            # 모멘텀(momentum): 이전 갱신 방향을 누적하여 현재 갱신에 반영하는 정도입니다.
            # 같은 방향의 기울기가 이어지면 이동을 가속하고,
            # 방향이 반복해서 바뀌는 구간에서는 진동을 줄이는 데 도움이 될 수 있습니다.
            # 0.0은 모멘텀을 사용하지 않는다는 뜻이며, 0.9 등이 자주 사용됩니다.
            momentum=0.0,

            # 가중치 감쇠(weight decay): 파라미터가 지나치게 커지는 것을 억제하는 규제입니다.
            # PyTorch의 SGD에서는 기울기에 "weight_decay × 현재 파라미터"를 더합니다.
            # 이는 L2 규제에 해당하며, 과적합을 줄이는 데 도움이 될 수 있습니다.
            # 0.0은 적용하지 않는다는 뜻이고, 너무 크게 설정하면 과소적합이 생길 수 있습니다.
            # 여기서는 model.parameters() 전체를 전달했으므로 편향에도 동일하게 적용됩니다.
            weight_decay=0.0
        )
    else:
        # AdamW 최적화기를 생성합니다.
        # 각 파라미터에 대해 기울기의 이동평균과 기울기 제곱의 이동평균을 관리하고,
        # 이를 이용해 파라미터별 갱신 크기를 조절합니다.
        # loss.backward()로 기울기를 계산한 뒤 optimizer.step()으로 갱신합니다.
        optimizer = torch.optim.AdamW(

            # 갱신할 모델의 파라미터를 전달합니다.
            # 현재 모델에서는 각 Linear 층의 가중치(weight)와 편향(bias)이 해당합니다.
            model.parameters(),

            # 학습률(learning rate): 파라미터 갱신 크기의 기본 배율입니다.
            # AdamW는 기울기의 이동평균과 제곱 이동평균을 이용하므로,
            # 단순히 "현재 기울기 × 학습률"만큼 이동하는 것은 아닙니다.
            # 같은 lr이라도 SGD와 AdamW의 실제 갱신 크기와 학습 결과는 다를 수 있습니다.
            # 너무 크면 학습이 불안정해지고, 너무 작으면 학습이 느려질 수 있습니다.
            lr=0.01,

            # 가중치 감쇠(weight decay): 파라미터가 지나치게 커지는 것을 억제하는 규제입니다.
            # AdamW는 손실의 기울기에 규제 항을 더하는 대신,
            # 적응적 기울기 갱신과 분리하여 파라미터를 직접 줄입니다.
            # 감쇠 부분만 보면 "파라미터 *= (1 - lr × weight_decay)"에 해당합니다.
            # 과적합을 줄이는 데 도움이 될 수 있지만, 너무 크면 과소적합이 생길 수 있습니다.
            # 0.0은 가중치 감쇠를 적용하지 않는다는 뜻입니다.
            # 양수로 설정하면 여기서는 전달한 모든 파라미터에 적용되므로 편향도 감쇠됩니다.
            weight_decay=0.0
            )

            # AdamW에는 SGD의 momentum 대신 betas라는 설정이 있습니다.
            # 생략하면 기본값 betas=(0.9, 0.999)를 사용합니다.
            # 첫 번째 값: 기울기의 이동평균에 이전 정보를 얼마나 유지할지 결정합니다.
            # 두 번째 값: 기울기 제곱의 이동평균에 이전 정보를 얼마나 유지할지 결정합니다.
            # 따라서 weight_decay=0.0이어도 과거 기울기를 활용하는 기능은 작동합니다.
        
    # [7] 모델 학습 (Training Loop) - 총 20회(Epoch) 반복
    for epoch in range(20):
        model.train()          # 모델을 '학습 모드'로 전환 (드롭아웃, 배치 정규화 등이 동작하도록 설정)
        optimizer.zero_grad()  # 새로운 반복마다 기존에 계산되어 누적되어 있던 기울기(Gradient)를 0으로 초기화
        
        # 순전파(Forward): 모델에 학습 데이터를 넣어 예측값을 구하고, 손실 함수로 정답과의 오차(loss) 계산
        loss = loss_fn(model(X_train), y_train)
        
        loss.backward()        # 역전파(Backward): 오차를 바탕으로 각 가중치(Weight)들의 기울기(Gradient) 계산
        optimizer.step()       # 가중치 갱신: 계산된 기울기와 설정한 알고리즘(SGD/AdamW) 규칙에 따라 가중치 업데이트
        
    # [8] 모델 평가 (Evaluation)
    model.eval()               # 모델을 '평가 모드'로 전환 (학습 시에만 필요한 기능들을 끔)
    with torch.no_grad():      # 평가 시에는 미분(기울기 계산)이 필요 없으므로 메모리 절약 및 연산 가속을 위해 그래디언트 추적을 비활성화
        # 검증 데이터를 모델에 넣어 손실을 계산하고, .item()을 통해 텐서에서 순수 파이썬 숫자(float)만 추출
        valid_loss = loss_fn(model(X_valid), y_valid).item()
        
    # [9] 결과 출력: 소수점 6자리까지 반올림하여 최종 검증 손실(Validation Loss) 비교
    print(name, "20 epochs / valid loss:", round(valid_loss, 6))

    """ 출력 결과 값
    SGD 20 epochs / valid loss: 0.484183
AdamW 20 epochs / valid loss: 0.307332  """