import torch
from torch import nn

"""
[RNN]
RNN 에 대한 설명은 생략 합니다.
아래 코드는 torch 에서 제공하는 RNN 레이어를 생성하여 순전파 하는 예시 입니다.
"""

input_size = 128
ouput_size = 256
num_layers = 3
bidirectional = True

model = nn.RNN(
    # 입력 데이터 사이즈
    input_size=input_size,
    # RNN 히든 레이어 사이즈 = 출력 데이터 사이즈
    hidden_size=ouput_size,
    # 순환 신경망 층수
    num_layers=num_layers,
    # 활성화 함수 종류 (relu, tanh)
    nonlinearity="tanh",
    # 입력 배치 크기를 첫번째 차원으로 사용할지 여부
    # True : [배치 크기, 시퀀스 길이, 입력 특성 크기]
    # False : [시퀀스 길이, 배치 크기, 입력 특성 크기]
    batch_first=True,
    # 양방향 순환 여부
    bidirectional=bidirectional,
)

batch_size = 4
sequence_len = 6

# 테스트를 위한 무작위 입력값 생성
inputs = torch.randn(batch_size, sequence_len, input_size)

# RNN 에 입력될 초기 은닉 상태
h_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size, ouput_size)

outputs, hidden = model(inputs, h_0)
print(outputs.shape)
print(hidden.shape)
