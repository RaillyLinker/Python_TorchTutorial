import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.single_layer_perceptron.main_model as single_layer_perceptron
import os
from torch import optim

"""
[단일 퍼셉트론]
- 퍼셉트론은, 동물의 신경망을 본뜬 알고리즘으로, 딥러닝의 인공 신경망의 가장 기본적인 형태입니다.
    입력을 받아 출력값을 반환하는 일반적인 선형 회귀 모델과 다른점은,
    신경망은 출력층 마지막에 선형 출력값을 비선형으로 바꿔주는 활성화 함수가 달려있다는 것입니다.
    신경 세포와 마찬가지로 활성화 함수로 출력된 출력값을 다른 퍼셉트론이 받고, 또 그 퍼셉트론의 신호를 다른 퍼셉트론들이 받는 것으로 이어져있는 구조입니다.
    아래 예시는 신경망 하나를 구현한 토치 모델을 만들어 사용하는 예시입니다.
    선형 함수에 활성화 함수를 Sigmoid 로 붙였으므로, 
    아래에 구현한 단일 퍼셉트론은 Logistic Regression 과 다를 것이 없습니다.
- 활성화 함수에는 Sigmoid 외에도 여러가지 방법이 존재합니다.
    일단 Sigmoid 함수는,
    def sigmoid(x): # 시그모이드 함수 정의
        return 1/(1+np.exp(-x))
    이렇게 구현이 가능 합니다.
    이렇게 하면 값이 커질수록 한없이 1 에 가까워지고, 값이 작아지면 한없이 0 에 가까워지는 S자형 결과값을 얻을 수 있습니다.
    여기서 0.5 를 기준으로 True(1), False(0) 를 나누면 분류 모델이 되는 것입니다. 
- 순전파 (Forward Propagation)
    인공 신경망에서 순전파란, 모델에 입력값을 넣어 예측을 하는 작업을 말합니다.
    ax + b 라는 모델로 구성된 신경망에, 2 라는 x 값이 입력되면, 2x + b 라는 값이 나오는 것입니다.
    인공 신경망으로 하는 머신 러닝, 즉 딥러닝의 목적은 순전파로 출력되는 값인 y 와, 실제 값인 collect_y 와의 차이를 줄이는 것입니다.
- 역전파 (Backpropagation)
    인공 신경망에 존재하는 여러 예측 모델의 파라미터를 학습 시킬 때는 오차 역전파라는 방식을 사용합니다.
    인공 신경망 모델이 예측 모델, 손실 함수가 존재한다고 하면, 손실 함수의 출력값을 줄이는 것이 목표겠죠?
    그렇다면 모델을 구성하는 각 파라미터 마다 이 최종적인 손실값에 영향을 끼치는 기울기를 구하고, 그에따라 파라미터를 수정해나가면 학습이 된 것입니다.
    수식을 설명하자면, 파라미터인 w 에 (학습률 * w 가 손실 값에 끼치는 영향) 을 빼주는 것입니다.
    
    예시를 통해 봅시다.
    
    1. 조건 :
        w = 3
        x = 2
        b = 1
        모델 = y = w * x + b
        정답 = yc = 4
        학습률 = 0.01
    2. 순전파 : 3 * 2 + 1 = 7
    3. 손실 계산 : (4 - 7)^2 = 4
    4. 기울기 계산 : 오차 역전파를 통해 기울기(gradient)를 계산하는 과정은 손실 함수의 가중치에 대한 미분을 통해 이루어집니다.
        손실 함수 출력값에 영향을 끼치는 파라미터인 w 와 b 의 각 미분을 구하면,
        wd = (2 * (y-yc) * x) = (2 * (7-4)*2) = 12
        bd = (2 * (y - yc)) = (2 * (7 - 4)) = 6
    5. 가중치 업데이트 : 기울기가 계산되었으니, 이제 이를 사용해 가중치 w와 편향 b를 업데이트할 수 있습니다.
        w = w - 학습률 * wd = (3 - 0.01 * 12) = 2.88
        b = b - 학습률 * bd = (1 - 0.01 * 6) = 0.94
        
    위와 같이 가중치를 업데이트 할 수 있습니다.
    
    위에서 미분 계산은 미분의 체인룰을 적용한 것입니다.
    
    y = f(g(x))
    와 같이 합성 함수가 존재할 때,
    
    y 에 대한 x 의 미분값은,
    y 에 대한 g 함수 결과값의 미분값 * g 함수의 결과값에 대한 x 의 미분값과 같습니다.
    
    이를 이용하여 존재하는 합성 함수의 미분 공식을 미리 만들어둘 수 있습니다.
    
    오차 역전파란 이름은, 손실 함수의 결과값을 뒤에서부터 앞쪽으로 순차적으로 계산하는 것을 의미합니다.
    
    모델의 함수를 순서대로 실행하여 입력값에서 출력값으로 계산을 진행하는 순전파와 비교하자면,
    모델 함수와 역방향인 모델 미분 함수를 손실값에서 차례로 계산을 진행하는 형태가 역전파라는 이름과 딱 맞는 것 같습니다.
    
    Torch 에서 제공해주는 Tensor 를 사용하여 계산을 진행하면, 위와 같은 역전파에 대한 로직을 구현할 것 없이 앞서도 사용해보았던
    
    loss.backward()
    
    라는 함수로 위와 같이 각 파라미터로 역전파하여 기울기를 구할 수 있는 것입니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # CSV 파일로 데이터셋 형성 (1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다.)
    # 여기서 X 값은 (True, True) 와 같이 Boolean 값이고, Y 값도 (True) 같은 Boolean 값입니다.
    # 즉, 이진분류를 해결하는 문제이자, 논리 회로를 구성하는 문제입니다.
    # 단순히 학습시키는 아래와 같은 예시에는 상관이 없지만, 일단 단층 신경망은 Xor 회로를 구현할 수 없습니다.
    # 다층 신경망부터 Xor 회로를 만들 수 있게 됩니다.
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../resources/datasets/perceptron.csv",
        x_column_labels=['x1', 'x2'],
        y_column_labels=['y1']
    )

    # 학습용, 검증용, 테스트용 데이터를 비율에 따라 분리
    train_dataset, validation_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.2
    )

    # 데이터 로더 래핑
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 생성
    model = single_layer_perceptron.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/single_layer_perceptron",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1000
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/single_layer_perceptron"
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

    # 모델 순전파
    with torch.no_grad():
        model.eval()
        inputs = torch.FloatTensor([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]).to(device)
        outputs = model(inputs)

        print("---------")
        print(outputs)
        print(outputs <= 0.5)


if __name__ == '__main__':
    main()
