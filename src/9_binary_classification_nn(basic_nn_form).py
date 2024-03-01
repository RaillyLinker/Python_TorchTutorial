import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.binary_classification.main_model as binary_classification
import os


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 모델 생성
    model = binary_classification.MainModel()

    # CSV 에서 데이터 가져오기 (ex : [{x : [x1, x2], y : [y1]}, {x : [x1, x2], y : [y1]}, ...])
    csv_data = tu.get_csv_file_data(
        csv_file_full_url="../_datasets/binary.csv",
        x_column_labels=['x1', 'x2', 'x3'],
        y_column_labels=['y1']
    )

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100., 82.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    dataset = tu.ModelDataset(input_data=csv_data)

    train_dataset, validation_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.2
    )

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.BCELoss(),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        learning_rate=0.0001,
        check_point_file_save_directory_path="../check_point_files/binary_classification",
        # check_point_load_file_full_path="../check_points/checkpoint(2024_02_29_17_51_09_330).pt"
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
