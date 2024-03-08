import torch
from torch import nn

"""
[배치 정규화 샘플]
(장점)
- 학습 속도 향상: 
    배치 정규화는 학습 속도를 향상시킬 수 있습니다. 
    그 이유는 각 레이어의 입력이 정규화되어 있기 때문에 학습 속도가 빨라지고, 따라서 더 빠르게 수렴할 수 있습니다.
- 초기화에 대한 덜 민감: 
    배치 정규화는 가중치 초기화에 대해 덜 민감합니다. 
    가중치 초기화가 잘못되었을 때도 학습이 잘 진행될 수 있습니다.
- 규제 효과: 
    배치 정규화는 일종의 규제(regularization) 효과를 가지며, 
    과적합을 방지하는 데 도움을 줄 수 있습니다.
- 더 큰 학습률 사용: 
    배치 정규화를 사용하면 더 큰 학습률을 사용할 수 있습니다. 
    이는 학습 과정을 가속화하고 더 빨리 수렴할 수 있게 합니다.
- 활성화 함수의 출력 분포를 조정: 
    배치 정규화는 활성화 함수의 출력 분포를 조정하여 기울기 소실(vanishing gradient) 문제를 완화합니다. 
    특히 시그모이드 함수와 같은 활성화 함수의 출력이 0 또는 1에 가까워질 때 발생하는 기울기 소실 문제를 완화하는 데 도움이 됩니다.

(배치 정규화 적용 시점)
- 배치 정규화는 보통 신경망의 각 레이어 이후에 배치됩니다. 
    주로 활성화 함수(예: ReLU) 이전이나 후에 배치 정규화를 적용하는 것이 일반적입니다.
- 선형 변환 이후: 
    대부분의 경우, 선형 변환(예: 완전 연결 레이어나 합성곱 레이어) 이후에 배치 정규화를 적용합니다. 
    이는 활성화 함수 이전에 배치 정규화를 적용하는 것이 일반적입니다.
- 활성화 함수 이전: 
    일반적으로 배치 정규화는 활성화 함수(예: ReLU) 이전에 배치됩니다. 
    이렇게 하면 활성화 함수의 입력 분포가 정규화되어 기울기 소실 문제를 완화할 수 있습니다.
- 완전 연결 레이어와 합성곱 레이어 사이: 
    배치 정규화는 보통 완전 연결 레이어나 합성곱 레이어 이후에 적용됩니다. 
    이 위치에서 배치 정규화를 사용하면 각 레이어의 출력이 다음 레이어로 전달되기 전에 정규화됩니다.
- 출력에 가까운 층들: 
    신경망의 출력에 가까운 층들에서도 배치 정규화를 사용하는 것이 일반적입니다. 
    이는 네트워크의 출력을 정규화하여 안정적인 예측을 만들어내도록 돕습니다.
- 요약하면, 배치 정규화는 보통 각 레이어의 선형 변환 이후 또는 활성화 함수 이전에 배치됩니다.
"""


# BatchNorm 구현
# 데이터의 평균을 0으로, 표준 편차를 1로 만듭니다. (데이터 분포를 표준 분포로 변경)
class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(CustomBatchNorm1d, self).__init__()
        # 가중치 및 편향 파라미터 생성
        self.gamma = nn.Parameter(torch.ones(num_features))  # scale parameter (학습 가능한 파라미터)
        self.beta = nn.Parameter(torch.zeros(num_features))  # shift parameter (학습 가능한 파라미터)

    def forward(self, model_in):
        # 배치 평균과 배치 분산 계산
        batch_mean = model_in.mean(dim=0)
        batch_var = model_in.var(dim=0, unbiased=False)

        # 배치 정규화 수행
        normalized_x = (model_in - batch_mean) / torch.sqrt(batch_var + 1e-5)

        # Scale 및 Shift 적용
        scaled_x = normalized_x * self.gamma + self.beta
        return scaled_x


# 배치 정규화 실행
x = torch.FloatTensor(
    [
        [-0.6577, -0.5797, 0.6360],
        [0.7392, 0.2145, 1.523],
        [0.2432, 0.5662, 0.322]
    ]
)

feature_size = x[0].shape[0]
print(feature_size)

# 커스텀 배치 정규화 실행
norm_tensor = CustomBatchNorm1d(feature_size)(x)
print(norm_tensor)

# 제공 배치 정규화 실행
norm_tensor = nn.BatchNorm1d(feature_size)(x)
print(norm_tensor)


# 적용 예시
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 선형 변환 이후, 활성화 이전 적용
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, model_in):
        model_out = model_in.view(-1, 28 * 28)
        model_out = self.fc1(model_out)
        model_out = self.bn1(model_out)
        model_out = self.relu1(model_out)
        model_out = self.fc2(model_out)
        return model_out
