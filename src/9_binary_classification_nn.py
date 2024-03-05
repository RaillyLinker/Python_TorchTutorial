import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.binary_classification.main_model as binary_classification
import os
from torch import optim

"""
[이진 분류 모델 - 토치 모델]
아래 코드는 이진 분류 모델을 본격적으로 토치 모델로 만들어 사용하는 예시를 보여줍니다.
앞서 작성한 NN 모델 템플릿대로 작성 하였습니다.
"""


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    dataset = tu.CsvModelDataset(
        csv_file_full_url="../_datasets/binary.csv",
        x_column_labels=['x1', 'x2', 'x3'],
        y_column_labels=['y1']
    )

    train_dataset, validation_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.2
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)

    # 모델 생성
    model = binary_classification.MainModel()

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
        for x, y in validation_dataloader:
            x = x.to(device)

            outputs = model(x)

            print(outputs)
            print(outputs >= torch.FloatTensor([0.5]).to(device))
            print("--------------------")


if __name__ == '__main__':
    main()
