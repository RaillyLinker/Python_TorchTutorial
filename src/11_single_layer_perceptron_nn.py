import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.single_layer_perceptron.main_model as single_layer_perceptron
import os
from torch import optim

"""
[단일 퍼셉트론]
퍼셉트론은, 동물의 신경망을 본뜬 알고리즘으로, 딥러닝의 인공 신경망의 가장 기본적인 형태입니다.
입력을 받아 출력값을 반환하는 일반적인 선형 회귀 모델과 다른점은,
신경망은 출력층 마지막에 선형 출력값을 비선형으로 바꿔주는 활성화 함수가 달려있다는 것입니다.
아래 예시는 신경망 하나를 구현한 토치 모델을 만들어 사용하는 예시입니다.
이것 단일로 사용하면 일반적인 분류 모델과 다를 것이 없습니다. (비선형적인 Xor 분류를 할 수 없음)
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../_datasets/perceptron.csv",
        x_column_labels=['x1', 'x2'],
        y_column_labels=['y1']
    )

    train_dataset, validation_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.2
    )

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
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
        check_point_file_save_directory_path="../check_point_files/binary_classification",
        # check_point_load_file_full_path="../check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt"
    )

    # 모델 저장
    model_file_save_directory_path = "../model_files/binary_classification"
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
