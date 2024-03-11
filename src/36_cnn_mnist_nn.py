import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.cnn_mnist_classifier.main_model as cnn_mnist_classifier
import os
from torch import optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms

"""
[합성곱 신경망 모델]
- 앞서 MNIST 손글씨 이미지 분류기를 만들어본 적이 있습니다.
    CNN 은 바로 그러한 이미지 분석에 효율적인 신경망 모델입니다.
    
- 이미지 데이터는 2 차원 데이터이면서, 데이터의 스칼라 값만큼 데이터의 배치 위치 정보도 중요한 데이터입니다.
    CNN 은 2차원 필터를 사용하여 데이터의 위치 정보까지 추출할 수 있습니다.
    사실, CNN 의 경우는 개인적으로 다른 글에서 정리를 많이 했었고, 여기서 정리하려면 꽤나 글이 길어지기에 설명을 생략하겠습니다.
    
- 아래는 합성곱 신경망 CNN 레이어를 사용하여 MNIST 손글씨 이미지 분류기를 만드는 예시입니다.
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
        transform=transforms.ToTensor(),
        download=True
    )

    validation_dataset = dsets.MNIST(
        root='../resources/datasets/MNIST_data/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = cnn_mnist_classifier.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.1),
        train_dataloader=train_dataloader,
        num_epochs=15,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/cnn_mnist_classifier",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/cnn_mnist_classifier"
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

"""
- 여기까지, 총 3 번을 MNIST 문제 해결 모델을 만들어 봤습니다.
    처음은, 기본적인 자동 학습 기능을 가진 단층 퍼셉트론으로 하여 0.273 정도의 loss 를 가지는 모델을,
    다음은 xor 문제에 대응 가능한 다층 인공 신경망 모델을 사용하여 0.076 정도의 loss 를 가지는 모델로 개선하였으며,
    이번에는 데이터의 위치 정보까지 활용하는 CNN 을 사용한 합성곱 신경망을 사용하여 0.021 정도의 loss 를 가지는 모델까지 발전시킬 수 있었습니다.
    비교적 단순하고 크기도 작은 MNIST 이미지가 아니라, 보다 크고 복잡한 이미지를 사용할 시에는 격차는 더욱 벌어질 것입니다.
    이처럼 결과를 보다시피 CNN 모델은 이미지 정보 분석에 매우 탁월한 성능을 보입니다.
    CNN 이 처음 주목을 받은 이후, 기존의 이미지 분석 모델들의 성능을 능가하기 시작하며,
    이후 계속해서 발전하여, 현재는 이미지 분류쪽으로는 기존 분석 모델은 물론, 사람의 인지 능력을 능가한다고 합니다.
    여러 개선 방식이 존재하겠지만, 결론적으로 CNN 레이어의 깊이가 깊어질수록 성능은 더더욱 나아집니다.
    (물론 파라미터가 많아지고 층이 깊어질수록 양질의 데이터가 필요하며, 학습시 확률적 요소가 커집니다.)
    지금은 이것을 사용하여 이미지 분류 문제에 사용하였지만, CNN 의 가장 큰 의의는 이미지 정보의 '압축'이라고 저는 생각합니다.
    특정 사진을 특정 객체라고 분류가 가능할 정도로 압축된 데이터와, 그 압축 모델은, 그 자체로 쓰이는 것 외에도,
    다른 모델에 특정 이미지의 정보를 입력하는 데이터 전처리기로서 사용하기 좋습니다.
    이렇게 잘 학습된 기존 CNN 분류 모델에서 인코딩 부분만을 떼어내어 다른 모델에 접합 시켜 사용하는 것을 전이학습이라 부르며,
    이 CNN 부분을 백본(BackBone) 모델이라고도 부릅니다.
"""
