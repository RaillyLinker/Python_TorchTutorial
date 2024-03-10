import torch
from torch import nn

"""
[장단기 메모리(Long Short-Term Memory, LSTM)]
- 앞서 배운 RNN을 가장 단순한 형태의 RNN이라고 하여 바닐라 RNN(Vanilla RNN)이라고 합니다. 
    바닐라 RNN 은 등장 직후 딥러닝을 이용한 자연어 처리에 대한 희망을 보여주었고, 바로 자연어 처리의 주류가 되었습니다. 
    이후 바닐라 RNN의 한계를 극복하기 위한 다양한 RNN의 변형이 나왔습니다. 
    LSTM도 그 중 하나입니다.
    
- 바닐라 RNN의 한계
    앞 챕터에서 바닐라 RNN은 출력 결과가 이전의 계산 결과에 의존한다는 것을 언급한 바 있습니다. 
    하지만 바닐라 RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있습니다. 
    RNN 정리글에도 설명하였듯, 바닐라 RNN의 시점(time step)이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생하기 때문입니다.
    이를 장기 의존성 문제(the problem of Long-Term Dependencies)라고 합니다.

- 아래 코드는 torch 에서 제공하는 LSTM 레이어를 생성하여 순전파 하는 예시 입니다.
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
