from pathlib import Path
import pandas as pd
import torch
from torch import nn

# 일관된 테스트 환경 구축 및 연산 제어
torch.set_num_threads(1)  # CPU 연산 시 단일 스레드를 사용하여 연산 결과의 재현성 확보
base = Path(__file__).resolve().parent  # 현재 실행 중인 파일의 절대 경로 폴더 지정

# 시각화 설정 (GUI 창을 띄우지 않고 메모리 상에서 PNG 파일로만 저장하도록 설정)
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# 데이터 로드 및 split 열의 기준에 따라 학습(train)/검증(valid) 데이터로 분할
df = pd.read_csv(base / "points.csv")
train = df[df["split"] == "train"]
valid = df[df["split"] == "valid"]

# Pandas 데이터프레임의 데이터를 딥러닝 연산에 최적화된 PyTorch 텐서(Tensor)로 변환
# 특성 데이터는 실수형(float32), 분류 클래스 번호인 정답(label)은 정수형(long)을 사용
X_train = torch.tensor(train[["x1", "x2"]].values, dtype=torch.float32)
y_train = torch.tensor(train["label"].values, dtype=torch.long)
X_valid = torch.tensor(valid[["x1", "x2"]].values, dtype=torch.float32)
y_valid = torch.tensor(valid["label"].values, dtype=torch.long)
# test 4건은 읽은 표에만 있으며 학습/평가 Tensor로 만들지 않습니다.

# 다중 분류 문제 해결을 위한 손실 함수 선언
loss_fn = nn.CrossEntropyLoss()

# 각 실험의 최종 결과 요약을 담을 리스트
summary = []

# 하이퍼파라미터인 학습률(lr)을 다르게 설정하여 baseline과 candidate 모델 성능 비교 실험 진행
for name, lr in [("baseline", 0.001), ("candidate", 0.01)]:
    # 매 실험마다 가중치 초기값을 똑같이 맞춰 공정한 비교를 할 수 있도록 랜덤 시드 고정
    torch.manual_seed(42)
    
    # 입력층(2) -> 은닉층(8) -> 출력층(2) 구조의 인공신경망 정의
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    
    # AdamW 옵티마이저 생성 및 매 실험 단계에 해당하는 학습률 주입
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    
    # 에포크별 손실 지표(Loss)의 변화를 실시간으로 기록할 리스트
    history = []
    
    # 1부터 20 에포크까지 반복 학습 및 평가 진행
    for epoch in range(1, 21):
        # --- 1. 모델 학습 모드 동작 ---
        model.train()
        optimizer.zero_grad()  # 새로운 미분 연산을 위해 기존 기울기(Gradient) 초기화
        
        # 현재 가중치 상태에서의 학습 데이터 손실값 계산
        loss = loss_fn(model(X_train), y_train)
        
        # 이미 zero_grad와 loss 계산이 끝났습니다. loss.backward() 다음 optimizer.step()을 호출하세요.
        # 학습 데이터만 사용합니다. 비교 후보마다 모델을 처음부터 새로 만듭니다.
        # >>> B08
        # TODO B08: 위 요구사항에 맞게 이 블록만 완성합니다.
        
        loss.backward()  # 역전파를 수행하여 가중치들의 기울기 계산
        
        # ⚠️ 매우 중요: 뒤에 양괄호()가 누락되면 함수가 실행되지 않아 가중치가 전혀 갱신되지 않습니다!
        optimizer.step()  # 계산된 기울기를 적용하여 가중치 업데이트 실행
        # <<< B08
        
        # --- 2. 모델 평가 모드 동작 ---
        model.eval()
        with torch.no_grad():  # 평가 단계에서는 연산 가속을 위해 불필요한 미분 트래킹 해제
            # 갱신 직후의 가중치로 학습 데이터와 검증 데이터의 순수 손실 값(.item())을 추출
            train_loss = loss_fn(model(X_train), y_train).item()
            valid_loss = loss_fn(model(X_valid), y_valid).item()
            
        # 한 epoch의 번호와 갱신 후 train_loss, valid_loss를 기록하세요.
        # history.append([epoch, train_loss, valid_loss]) 형태로 추가합니다.
        # >>> B09
        # TODO B09: 위 요구사항에 맞게 이 블록만 완성합니다.
        
        # DataFrame 생성을 위해 리스트 형태로 에포크별 지표를 누적 저장
        history.append([epoch, train_loss, valid_loss])
        # <<< B09
        
    # 데이터 분석 및 파일 저장 과정
    # 20에포크 동안 기록된 history 데이터를 Pandas 데이터프레임으로 구축
    table = pd.DataFrame(history, columns=["epoch", "train_loss_eval", "valid_loss"])
    table.to_csv(base / f"history_{name}.csv", index=False)  # 결과를 CSV 파일로 기록
    
    # 검증 손실(valid_loss)이 최소가 되는 지점의 행(Row) 정보 추출
    best = table.loc[table["valid_loss"].idxmin()]
    
    # 종합 분석 요약 리스트에 [실험명, 학습률, 최적에포크번호, 최소검증손실] 추가
    summary.append([name, lr, int(best["epoch"]), best["valid_loss"]])
    
    # --- 각 실험별 학습 곡선(Learning Curve) 시각화 및 이미지 저장 ---
    plt.figure(figsize=(8, 5))
    plt.plot(table["epoch"], table["train_loss_eval"], label="train (eval mode)")
    plt.plot(table["epoch"], table["valid_loss"], label="valid")
    plt.axvline(best["epoch"], linestyle="--", label="best epoch")  # 최적 지점에 수직 점선 표시
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(base / f"curves_{name}.png", dpi=150)  # 그래프 이미지 파일 저장
    plt.close()

# --- 최종 실험 비교 결과 분석 및 리포트 테이블 작성 ---
result = pd.DataFrame(summary, columns=["setting", "lr", "best_epoch", "best_valid_loss"])
result.to_csv(base / "experiment_table.csv", index=False)  # 실험 대조표를 CSV 파일로 저장

# 인덱스를 제외하고 보기 좋게 콘솔 창에 실험 결과 출력
print(result.round(6).to_string(index=False))
print("test 미사용 / best epoch는 관찰한 최소 지점, 기본필수에서는 모델 저장·복원하지 않음")