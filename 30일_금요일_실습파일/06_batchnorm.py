import torch
from torch import nn

# 가중치 초기화 및 일관된 결과를 확인하기 위해 랜덤 시드를 고정합니다.
torch.manual_seed(42)

# (4행, 2열) 크기의 입력 텐서 X를 생성합니다. (샘플 데이터 4개, 특성 개수 2개)
X = torch.tensor([[1.,2.],[2.,1.],[3.,4.],[4.,3.]])

# 배치 정규화층 생성: 입력되는 특성(Feature)의 개수가 8개이므로 num_features=8로 설정합니다.
bn = nn.Dropout(p=0.5) # (위 선언은 아래 nn.Sequential 안에서 활용됩니다)
bn = nn.BatchNorm1d(num_features=8)

# 선형 레이어(2->8) -> 배치 정규화(8) -> ReLU 활성화 함수로 이어지는 심층 신경망 모델 정의
model = nn.Sequential(nn.Linear(2, 8), bn, nn.ReLU())

# --- 1. 학습 모드 (Training Mode) ---
# 학습 모드일 때 BatchNorm1d는 입력되는 미니배치 데이터의 평균과 분산을 
# 실시간으로 계산하여 누적 평균(running_mean)과 누적 분산(running_var)을 지속적으로 업데이트(갱신)합니다.
model.train()

# model에 X를 넣어 output을 구하세요. 입력 (4,2)는 Linear에서 (4,8)이 됩니다.
# BatchNorm1d(8)은 8개 특성의 통계를 유지합니다.
# >>> B07
# TODO B07: 위 요구사항에 맞게 이 블록만 완성합니다.

# 입력 데이터 X(4,2)를 모델에 통과시켜 최종 출력값 output(4,8)을 연산합니다.
# 이 과정에서 배치 정규화층(bn)의 누적 통계치가 처음으로 계산되고 갱신됩니다.
output = model(X)
# <<< B07

# 데이터 크기 출력 확인: 선형 변환을 거쳐 특성이 2개에서 8개로 늘어났음을 확인 가능합니다.
print("입력:", tuple(X.shape), "출력:", tuple(output.shape))

# bn.running_mean: 배치 정규화가 학습을 진행하며 기록한 '8개 특성 각각의 누적 평균값'의 형태를 출력합니다.
print("running_mean:", tuple(bn.running_mean.shape))

# .clone(): 현재 학습 모드에서 갱신된 누적 평균값 배열을 복사하여 'before' 변수에 따로 보관합니다.
before = bn.running_mean.clone()

# --- 2. 평가 모드 (Evaluation Mode) ---
# 평가 모드일 때 BatchNorm1d는 학습 중에 모아둔 누적 평균과 분산을 고정한 채 
# 오직 데이터를 정규화하는 데만 사용합니다. 즉, 새로운 데이터를 넣어도 통계치가 더 이상 변하지 않습니다.
model.eval()

# 평가 단계에서는 미분(기울기) 계산이 불필요하므로 메모리 보존을 위해 그라디언트 추적을 비활성화합니다.
with torch.no_grad():
    # 평가 모드 상태인 모델에 다시 한번 데이터 X를 입력해 봅니다.
    model(X)

# torch.equal(A, B): 두 텐서의 원소가 완벽히 일치하면 True, 다르면 False를 반환합니다.
# 평가 모드에서 데이터를 다시 흘려보냈음에도, 학습 종료 직전 기록했던 통계치(before)가 
# 전혀 변하지 않고 그대로 유지되었음을 증명(True 출력)하는 문장입니다.
print("eval에서 통계 유지:", torch.equal(before, bn.running_mean))

""" 출력 결과 값
입력: (4, 2) 출력: (4, 8)
running_mean: (8,)
eval에서 통계 유지: True """