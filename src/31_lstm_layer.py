import torch
from torch import nn

"""
[LSTM]
LSTM 에 대한 설명은 생략 합니다.
아래 코드는 torch 에서 제공하는 LSTM 레이어를 생성하여 순전파 하는 예시 입니다.
동일한 구조인 GRU 역시 아래와 같은 방식으로 사용하면 됩니다.
"""

input_size = 128
ouput_size = 256
num_layers = 3
bidirectional = True
proj_size = 64

model = nn.LSTM(
    input_size=input_size,
    hidden_size=ouput_size,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    proj_size=proj_size,
)

batch_size = 4
sequence_len = 6

inputs = torch.randn(batch_size, sequence_len, input_size)
h_0 = torch.rand(
    num_layers * (int(bidirectional) + 1),
    batch_size,
    proj_size if proj_size > 0 else ouput_size,
)
c_0 = torch.rand(num_layers * (int(bidirectional) + 1), batch_size, ouput_size)

outputs, (h_n, c_n) = model(inputs, (h_0, c_0))

print(outputs.shape)
print(h_n.shape)
print(c_n.shape)
