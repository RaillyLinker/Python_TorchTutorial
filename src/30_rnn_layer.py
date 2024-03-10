import torch
from torch import nn

"""
[RNN]
- 순환 신경망(Recurrent Neural Network, RNN)
    RNN(Recurrent Neural Network)은 시퀀스(Sequence) 모델입니다. 
    입력과 출력을 시퀀스 단위로 처리하는 모델입니다. 
    번역기를 생각해보면 입력은 번역하고자 하는 문장. 즉, 단어 시퀀스입니다. 
    출력에 해당되는 번역된 문장 또한 단어 시퀀스입니다. 
    이러한 시퀀스들을 처리하기 위해 고안된 모델들을 시퀀스 모델이라고 합니다. 
    그 중에서도 RNN은 딥 러닝에 있어 가장 기본적인 시퀀스 모델입니다.
    
- 용어는 비슷하지만 순환 신경망과 재귀 신경망(Recursive Neural Network)은 전혀 다른 개념입니다.  

- 아래 코드는 torch 에서 제공하는 RNN 레이어를 생성하여 순전파 하는 예시 입니다.
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
