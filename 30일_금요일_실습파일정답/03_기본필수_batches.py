import torch
from torch.utils.data import TensorDataset, DataLoader

# 학습용 공통 데이터와 별개인 '배치 개수만 확인하는' 20개 숫자입니다.
X = torch.arange(40, dtype=torch.float32).reshape(20, 2)
y = torch.arange(20) % 2
dataset = TensorDataset(X, y)
# DataLoader(dataset, batch_size=6, shuffle=False)를 만드세요.
# 20개를 순서대로 나누면 6·6·6·2개입니다. dataset은 필수, batch_size와 shuffle은 선택 인자입니다.
# >>> B04
# 해설 B04: 아래 구현은 위 요구사항에 대응합니다.
loader = DataLoader(dataset, batch_size=6, shuffle=False)   # train 할때와 valid 할때 다르게 설정해줘야해요
# <<< B04
for iteration, (xb, yb) in enumerate(loader, start=1):
    print("iteration:", iteration, "X:", tuple(xb.shape), "y:", tuple(yb.shape))
print("한 epoch의 iteration:", len(loader))
