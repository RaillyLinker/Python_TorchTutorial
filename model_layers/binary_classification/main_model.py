from torch import nn


class MainModel(nn.Module):
    """
    description : 선형 회귀 모델
    input_shape : [[x1, x2, x3], [x1, x2, x3], ...]
    output_shape : [[y1], [y1], ...]
    """

    def __init__(self):
        super().__init__()
        # 모델 내 레이어
        self.layer = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, model_in):
        model_out = self.layer(model_in)
        return model_out
