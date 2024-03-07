import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.multi_layer_perceptron.main_model as multi_layer_perceptron
import os
from torch import optim

"""
[다중 퍼셉트론]
단일 퍼셉트론을 여러개 쌓은 형태의 모델입니다.
단일 퍼셉트론을 선 하나로 생각하세요. (선 하나를 그어두고 그 위에 있는 것은 1, 아래에 있는 것은 0으로 분류)
단일 퍼셉트론 끝에 달린 활성화 함수로 모델의 선형성을 끊었는데,
이렇게 끊어진 선들이 결합됨으로써 선형적으로는 판별할 수 없는 여러 문제들을 해결 할 수 있습니다. (Xor 문제 등)
한붓 그리기에서 일반적인 그리기로 업그레이드 된 것입니다.
특히나 퍼셉트론은 데이터와 정답이 있다면 내부적으로 자동으로 학습을 하기에 이 선들이 많다면 세상에 풀지 못할 문제가 없습니다.
이것이 기본 딥러닝 모델의 원리인데, 결국 딥러닝 개발이라는 것은, 이러한 퍼셉트론을 어떻게 쌓을 것인지에 대한 문제라고 볼 수 있습니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    # CSV 파일로 데이터셋 형성 (1 행에 라벨이 존재하고, 그 라벨로 x, y 데이터를 분류 합니다.)
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
    model = multi_layer_perceptron.MainModel()

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCELoss(),
        optimizer=optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        check_point_file_save_directory_path="../_check_point_files/multi_layer_perceptron",
        # check_point_load_file_full_path="../_check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        log_freq=1000
    )

    # 모델 저장
    model_file_save_directory_path = "../_torch_model_files/multi_layer_perceptron"
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
