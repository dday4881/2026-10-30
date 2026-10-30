import torch
from torch.utils.data import TensorDataset, DataLoader

# 시스템 설정 (1개 스레드 고정)은 이전 파일들과 동일하게 일관된 테스트 환경을 만듭니다.
torch.set_num_threads(1)

# 학습용 공통 데이터와 별개인 '배치 개수만 확인하는' 20개 숫자입니다.
# 0부터 39까지의 숫자 40개를 생성한 뒤, (20행, 2열) 크기의 2차원 텐서로 변형합니다. (데이터 20개, 특성 2개)
X = torch.arange(40, dtype=torch.float32).reshape(20, 2)

# 0부터 19까지의 숫자를 2로 나눈 나머지(0 또는 1)로 정답 레이블 20개를 만듭니다. (정수형)
y = torch.arange(20) % 2

# TensorDataset: 입력 데이터(X)와 정답 데이터(y)를 하나로 묶어주는 팩(Dataset)을 만듭니다.
dataset = TensorDataset(X, y)

# DataLoader(dataset, batch_size=6, shuffle=False)를 만드세요.
# 20개를 순서대로 나누면 6·6·6·2개입니다. dataset은 필수, batch_size와 shuffle은 선택 인자입니다.
# >>> B04
# TODO B04: 위 요구사항에 맞게 이 블록만 완성합니다.

# DataLoader: 데이터셋을 미니배치 크기만큼 쪼개서 모델에 순차적으로 공급해 주는 역할을 합니다.
# batch_size=6: 한 번에 6개씩 묶어서 가져옵니다.
# shuffle=False: 데이터를 섞지 않고 순서대로 가져옵니다.
loader = DataLoader(dataset, batch_size=6, shuffle=False)  # train 할때와 valid 할때 다르게 설정해줘야해요
# <<< B04

# 반복문(Loop)을 돌며 배치 데이터를 하나씩 꺼내어 형태를 확인합니다.
# enumerate(..., start=1)을 사용하여 몇 번째 반복(iteration)인지 1부터 세어줍니다.
for iteration, (xb, yb) in enumerate(loader, start=1):
    # xb.shape와 yb.shape는 각각 현재 미니배치의 입력 데이터 크기와 정답 데이터 크기입니다.
    # 예: 첫 번째 데이터는 (6, 2) 크기의 입력과 (6,) 크기의 정답이 나옵니다.
    print("iteration:", iteration, "X:", tuple(xb.shape), "y:", tuple(yb.shape))

# len(loader): 전체 데이터(20개)를 배치 크기(6개)로 나누었을 때 나오는 총 묶음(배치)의 개수입니다.
# 20 / 6 = 3.33이므로 반올림이 아닌 올림 처리가 되어 총 4개의 묶음이 생깁니다. (6개, 6개, 6개, 나머지 2개)
print("한 epoch의 iteration:", len(loader))

""" 출력 결과 값
iteration: 1 X: (6, 2) y: (6,)
iteration: 2 X: (6, 2) y: (6,)
iteration: 3 X: (6, 2) y: (6,)
iteration: 4 X: (2, 2) y: (2,)
한 epoch의 iteration: 4 """