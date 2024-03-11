import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.multi_layer_perceptron_for_mnist.main_model as multi_layer_perceptron_for_mnist
import os
from torch import optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

"""
[다층 퍼셉트론으로 MNIST 분류하기]
- 앞서 소프트맥스 회귀로 MNIST 데이터를 분류하는 실습을 해봤습니다. 
    소프트맥스 회귀 또한 인공 신경망이라고 볼 수 있는데, 
    입력층과 출력층만 존재하므로 소프트맥스 함수를 활성화 함수로 사용한 '단층 퍼셉트론'이라고 할 수 있습니다. 
    이번 챕터에서는 은닉층을 추가로 넣어 다층 퍼셉트론을 구현하고, 딥 러닝을 통해서 MNIST 데이터를 분류해봅시다.
    
- 아래 코드에서 기존의 MINST 회귀 코드와 다른 것은, 
    기존에는 레이어가 출력 레이어 단 하나뿐이었지만, 지금은 출력 레이어와 합쳐 총 3개의 레이어로 이루어진 모델을 사용한다는 것 뿐입니다.
    실제로 두 모델의 학습 결과를 확인하면,
    기존 모델의 loss 가 0.273 정도라면, 이번 모델의 loss 는 0.076 정도로 매우 낮아진 것을 볼 수 있습니다.
    또한 상황에 따라 다르겠지만, training loss 와 validate loss 간의 차이가 커진 것도 볼 수 있을 것입니다.
    즉, xor 을 포함하여 더 복잡한 문제를 풀 수 있는 딥 뉴럴 네트워크 모델의 성능과 그 학습시 과적합(Overfitting)의 문제를 여실히 확인 할 수 있습니다.
- 저번 코드와의 비교 결과로 인하여 다층 신경망 모델에서 학습시 loss 와 검증시 loss 간의 차이가 벌어질 수 있다는 것을 보았습니다.
    머신러닝 모델은 학습을 많이 하는 것이 중요하죠. 그렇기에 빅데이터의 중요성이 부각되는 것입니다.
    그러나 동일한 데이터를 사용하고 또 사용한다면, 해당 데이터에만 특화된 모델이 만들어지는 것입니다.
    해당 데이터에만 존재하는 노이즈나 특징들을 더 중요하게 학습하게 되는 것이죠.(사람도 같은 일만 계속하면 문제를 해결한다기보다는 작업량을 줄이는 형태로 꼼수를 쓰죠.)
    학습시에는 loss 가 낮은데, 학습한 적이 없는 검증, 혹은 실제 예측시 loss 가 높아지는 현상이 일어나게 되며, 이를 과적합이라 합니다.
- 과적합을 막는 방법은,
    1. 양질의 학습 데이터를 사용하고, 학습한 데이터에 대한 재활용을 줄일 것
    2. 모델의 복잡도를 줄일 것
    3. 가중치 규제(Regularization) 을 적용할 것
    4. 드롭아웃을 사용할 것
    
    위와 같습니다.
    각 방법에 대한 설명과 적용 방식은 다른 글에 정리하겠습니다.
"""


# [MNIST 분류]
def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # x : (28, 28), y : (1) 형식입니다. y 의 경우는 One Hot Encoding 없이 CrossEntropyLoss 함수에 그대로 넣어줘도 됩니다.
    train_dataset = dsets.MNIST(
        root='../resources/datasets/MNIST_data/',
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: x.view(-1)  # 이미지를 1차원 벡터로 평탄화
            ]
        ),
        download=True
    )

    validation_dataset = dsets.MNIST(
        root='../resources/datasets/MNIST_data/',
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                lambda x: x.view(-1)  # 이미지를 1차원 벡터로 평탄화
            ]
        ),
        download=True
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = multi_layer_perceptron_for_mnist.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        train_dataloader=train_dataloader,
        num_epochs=15,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/multi_layer_perceptron_for_mnist_nn",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/multi_layer_perceptron_for_mnist_nn"
    if not os.path.exists(model_file_save_directory_path):
        os.makedirs(model_file_save_directory_path)
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path=model_file_save_directory_path
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)


if __name__ == '__main__':
    main()
