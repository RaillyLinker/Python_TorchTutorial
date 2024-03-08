from torch import nn

"""
[모델에서 가중치를 초기화 하는 방법]
가중치를 초기화 한 모델 사용 방법입니다.
앞으로 모델을 작성할 때에는 아래와 같이 초기화를 설정하는 것이 좋습니다.
"""


class MainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(20, 10),
            nn.Sigmoid()
        )
        self.layer3 = nn.Linear(in_features=10, out_features=1)

        self._init_weights()

    def _init_weights(self):
        # ReLU 를 활성화 함수로 사용하는 layer1 에는 Kaiming / He 초기화를 적용
        nn.init.kaiming_uniform_(self.layer1[0].weight, mode='fan_in', nonlinearity='relu')
        self.layer1[0].bias.data.fill_(0)

        # 그 외에는 Xavier 초기화를 적용
        nn.init.xavier_uniform_(self.layer2[0].weight)
        self.layer2[0].bias.data.fill_(0.01)

        nn.init.xavier_uniform_(self.layer3.weight)
        self.layer3.bias.data.fill_(0.01)

        # RNN, LSTM, GRU 등의 재귀적 모델에는 직교 초기화 (Orthogonal Initialization 을 사용하세요.)

    def forward(self, model_in):
        model_out = self.layer1(model_in)
        model_out = self.layer2(model_out)
        model_out = self.layer3(model_out)
        return model_out
