from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 메모리 상에서 PNG 파일로 바로 저장하도록 설정합니다.
import matplotlib.pyplot as plt

# 아래 20개 값은 학습곡선 읽기용으로 제공한 설명용 값입니다.
# 실제 모델 실행 결과라고 보고하지 않습니다. 07번은 직접 학습한 곡선을 그립니다.
# 에포크가 진행됨에 따라 학습 데이터의 손실(오차)은 끊임없이 계속 줄어듭니다.
train_losses = [0.90,0.76,0.65,0.56,0.49,0.43,0.38,0.33,0.29,0.25,
                0.22,0.20,0.18,0.16,0.14,0.13,0.12,0.11,0.10,0.09]

# 반면 검증 손실은 어느 순간(여기서는 11번째)까지만 줄어들다가, 
# 그 이후에는 모델이 학습 데이터에만 너무 과하게 맞춰지면서(과적합, Overfitting) 오히려 다시 상승합니다.
valid_losses = [0.94,0.81,0.71,0.61,0.53,0.47,0.42,0.38,0.35,0.33,
                0.32,0.33,0.35,0.37,0.40,0.43,0.46,0.48,0.50,0.52]

# valid_losses에서 가장 작은 값의 위치를 구하고 1을 더해 best_epoch에 넣으세요.
# 리스트 위치는 0부터, 여기서 출력할 epoch는 1부터 시작합니다.
# >>> B05
# TODO B05: 위 요구사항에 맞게 이 블록만 완성합니다.
# 해설 B05: 아래 구현은 위 요구사항에 대응합니다.
# 검증 손실(Validation Loss)이 가장 낮은 최적의 에포크(Epoch) 번호를 계산합니다.
# 1. min(valid_losses): 에포크별 검증 손실 리스트에서 가장 작은 최솟값(최적의 성능)을 찾습니다.
# 2. .index(...): 그 최솟값이 리스트의 몇 번째 인덱스(0부터 시작)에 위치하는지 구합니다.
# 3. + 1: 인덱스는 0부터 시작하지만 실제 에포크 번호는 1부터 시작하므로 1을 더해 보정합니다.

# min()으로 가장 작은 오차인 0.32를 찾고, index()로 그 위치가 10번째 방임을 알아낸 뒤, 1을 더해 11번째 에포크임을 구합니다.
best_epoch = valid_losses.index(min(valid_losses)) + 1
# <<< B05

# 최적의 에포크 번호와 그때의 최소 검증 손실 값을 콘솔에 출력합니다.
print("best epoch:", best_epoch, "최소 valid loss:", min(valid_losses))

# --- 학습 곡선 시각화 그래프 그리기 ---
plt.figure(figsize=(8, 5)) # 가로 8인치, 세로 5인치 크기의 그래프 창 생성

# X축을 1부터 20까지로 설정하여 에포크에 따른 train_loss와 valid_loss를 선으로 표현합니다.
plt.plot(range(1, 21), train_losses, label="train")
plt.plot(range(1, 21), valid_losses, label="valid")

# 모델의 성능이 가장 좋은 최적의 지점(best_epoch)에 세로 점선(--)을 그어 시각적으로 강조합니다.
plt.axvline(best_epoch, linestyle="--", label="best epoch")

plt.xlabel("Epoch")               # X축 이름표 설정
plt.ylabel("Cross entropy loss")  # Y축 이름표 설정
plt.legend()                      # 선들이 무엇을 뜻하는지 범례(우측 상단 상자) 표시
plt.tight_layout()                # 그래프 구성 요소들이 겹치지 않게 여백을 자동으로 조절

# 현재 파이썬 파일이 있는 폴더 안에 'practice04_curves.png'라는 이름으로 그래프를 고화질(dpi=150) 저장합니다.
plt.savefig(Path(__file__).resolve().parent / "practice04_curves.png", dpi=150)
plt.close() # 메모리 확보를 위해 그린 그래프 창을 닫아줍니다.

print("생성: practice04_curves.png")

""" 출력 결과 값
best epoch: 11 최소 valid loss: 0.32
생성: practice04_curves.png(파일 경로는 실습04 png) """