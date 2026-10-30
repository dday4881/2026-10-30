import torch
from torch import nn

torch.manual_seed(42)
X = torch.ones(2, 4)
dropout = nn.Dropout(p=0.5)
dropout.train()
for i in range(3):
    print("train", i + 1, dropout(X).tolist())
# Dropout을 평가 모드로 바꾸세요. dropout.eval()은 mode만 바꿉니다.
# 평가 모드의 Dropout은 입력을 그대로 전달하며, gradient 비활성화와는 별개의 기능입니다.
# >>> B06
# 해설 B06: 아래 구현은 위 요구사항에 대응합니다.
dropout.eval() #dropout.train(mode=False) 이렇게 써도 똑같습니다. 하지만 평가할때는 eval로 써주세요
# <<< B06
for i in range(3):
    print("eval", i + 1, dropout(X).tolist())
