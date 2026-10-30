from pathlib import Path
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

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
    for epoch in range(20):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        valid_loss = loss_fn(model(X_valid), y_valid).item()
    print(name, "20 epochs / valid loss:", round(valid_loss, 6))
