import torch
from torch import nn

"""
[합성곱 신경망 모델]
아래는 합성곱 신경망 CNN 레이어를 사용하여 모델을 만드는 예시입니다.
추가로, Pooling 의 경우에는 실제 데이터의 손실을 일으키기에 최근에는 Conv 계층의 간격을 조정하여 연산량을 줄이는 방법을 사용한다고 합니다.
"""


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 매트릭스 데이터를 벡터 데이터로 변경하여 이미지 정보를 담은 벡터로 인코딩 하는 CNN 출력 레이어
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x)
        x = self.fc(x)
        return x
