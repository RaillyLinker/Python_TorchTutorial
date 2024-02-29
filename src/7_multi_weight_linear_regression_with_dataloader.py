import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

# 학습 독립 변수 데이터
train_x = torch.FloatTensor(
    [
        [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]
    ]
)

# 학습 독립 변수 데이터
train_y = torch.FloatTensor(
    [
        [0.1, 1.5], [1, 2.8], [1.9, 4.1], [2.8, 5.4], [3.7, 6.7], [4.6, 8]
    ]
)

# 학습 데이터셋 객체
train_dataset = TensorDataset(train_x, train_y)

# 학습 데이터 로더 객체 (배치 사이즈, 셔플 여부, 배치 분할 후 여분 데이터 버릴지 여부)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# 선형 회귀 모델
model = nn.Linear(in_features=2, out_features=2, bias=True)

# 손실 함수
criterion = nn.MSELoss()

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=0.001)

for epoch in range(20000):
    # 에폭별 손실값
    epoch_loss = 0.0

    # 데이터 로더에서 배치 데이터 가져오기
    for batch in train_dataloader:
        x, y = batch

        # 모델 순전파
        model_out = model(x)

        # 비용 함수 계산
        model_loss = criterion(model_out, y)

        # 옵티마이저 초기화
        optimizer.zero_grad()

        # 비용 함수 결과 역전파 = 옵티마이저 기울기 계산
        model_loss.backward()

        # 계산된 기울기로 파라미터(weight, bias) 수정
        optimizer.step()

        epoch_loss += model_loss

    epoch_loss = epoch_loss / len(train_dataloader)

    if (epoch + 1) % 1000 == 0:
        print(f"Epoch : {epoch + 1:4d}, Model : {list(model.parameters())}, Cost : {epoch_loss:.3f}")
