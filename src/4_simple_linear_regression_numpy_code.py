import numpy as np

"""
[선형 회귀 모델 - 넘파이]
아래 코드는 머신러닝의 대표적인 두가지 문제인 회귀 문제, 분류 문제 중 회귀 문제에 속하는 가장 단순한 선형 회귀 모델을
토치 라이브러리의 지원을 받지 않고 넘파이로 구현한 것입니다.
선형 회귀 모델의 기본 형태와 가중치 수정을 통한 머신 러닝 학습 방식을 확인 할 수 있습니다.
"""

# 학습용 독립 변수 데이터
train_x = np.array(
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
     [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
     [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]]
)

# 학습용 종속 변수 데이터
train_y = np.array(
    [[0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
     [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
     [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]]
)

# 선형 회귀 모델(y = (x * w) + b) 를 이루는 학습 가능한 파라미터(w, b)
model_weight = 0.0
model_bias = 0.0

# 에포크 (학습 횟수)
epoch_n = 10000
# 학습률
learning_rate = 0.005

for epoch in range(epoch_n):
    # 선형 회귀 모델 순전파
    model_out = model_weight * train_x + model_bias

    # 예측 실패 정도 측정용 손실 함수(MSE)
    model_loss = ((train_y - model_out) ** 2).mean()

    # 1 에폭마다 모델 파라미터 수정
    model_weight = model_weight - learning_rate * ((model_out - train_y) * train_x).mean()
    model_bias = model_bias - learning_rate * (model_out - train_y).mean()

    if (epoch + 1) % 1000 == 0:
        # 1000 에폭마다 로깅
        print(f"Epoch : {epoch + 1:4d}, Weight : {model_weight:.3f}, Bias : {model_bias:.3f}, Cost : {model_loss:.3f}")
