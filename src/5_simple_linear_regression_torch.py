import torch
from torch import optim

# 학습 독립 변수 데이터
train_x = torch.FloatTensor(
    [
        [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15], [16], [17], [18], [19], [20],
        [21], [22], [23], [24], [25], [26], [27], [28], [29], [30]
    ]
)

# 학습 종속 변수 데이터
train_y = torch.FloatTensor(
    [
        [0.94], [1.98], [2.88], [3.92], [3.96], [4.55], [5.64], [6.3], [7.44], [9.1],
        [8.46], [9.5], [10.67], [11.16], [14], [11.83], [14.4], [14.25], [16.2], [16.32],
        [17.46], [19.8], [18], [21.34], [22], [22.5], [24.57], [26.04], [21.6], [28.8]
    ]
)

# 모델 가중치
model_weight = torch.zeros(1, requires_grad=True)

# 모델 편차
model_bias = torch.zeros(1, requires_grad=True)

# 옵티마이저
optimizer = optim.SGD([model_weight, model_bias], lr=0.001)

for epoch in range(10000):
    # 선형 회귀 모델 순전파
    model_out = train_x * model_weight + model_bias

    # 비용 함수 (MSE) 실행
    model_loss = torch.mean((model_out - train_y) ** 2)

    # 옵티마이저 기울기 초기화
    optimizer.zero_grad()

    # 비용 함수 결과 역전파 = 옵티마이저 기울기 계산
    model_loss.backward()

    # 계산된 기울기로 파라미터(weight, bias) 수정
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(
            f"Epoch : {epoch + 1:4d}, Weight : {model_weight.item():.3f}, Bias : {model_bias.item():.3f}, Cost : {model_loss:.3f}")
