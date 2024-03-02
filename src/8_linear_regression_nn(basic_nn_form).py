import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.linear_regression.main_model as linear_regression
import os


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 모델 생성
    model = linear_regression.MainModel()

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
    dataset = tu.ModelDataset(
        csv_file_full_url="../_datasets/linear.csv",
        x_column_labels=['x1', 'x2'],
        y_column_labels=['y1']
    )

    train_dataset, validation_dataset, test_dataset = tu.split_dataset(
        dataset=dataset,
        train_data_rate=0.8,
        validation_data_rate=0.1,
        test_data_rate=0.1
    )

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=10, shuffle=True, drop_last=True)

    # 모델 학습
    tu.train_model(
        device=device,
        model=model,
        criterion=nn.MSELoss(),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        learning_rate=0.0001,
        check_point_file_save_directory_path="../check_point_files/linear_regression",
        # check_point_load_file_full_path="../check_point_files/~/checkpoint(2024_02_29_17_51_09_330).pt",
        weight_decay=0.001,
        momentum=0.9
    )

    # 모델 저장
    model_file_save_directory_path = "../model_files/linear_regression"
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

    # 불러온 모델 순전파
    with torch.no_grad():
        model.eval()
        inputs = torch.FloatTensor(
            [
                [1 ** 2, 1],
                [5 ** 2, 5],
                [11 ** 2, 11]
            ]
        ).to(device)
        outputs = model(inputs)
        print(outputs)


if __name__ == '__main__':
    main()
