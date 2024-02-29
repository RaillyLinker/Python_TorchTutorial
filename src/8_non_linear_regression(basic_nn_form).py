import torch
from torch import nn
from torch.utils.data import DataLoader
import utils.torch_util as tu
import model_layers.non_linear_regression.main_model as non_linear_regression


def main():
    # 사용 가능 디바이스
    device = tu.get_gpu_support_device(gpu_support=True)

    # 모델 생성
    model = non_linear_regression.MainModel()

    # CSV 에서 데이터 가져오기 (ex : [{x : [x1, x2], y : [y1]}, {x : [x1, x2], y : [y1]}, ...])
    csv_data = tu.get_csv_file_data(
        csv_file_full_url="../_datasets/non_linear.csv",
        x_column_labels=['x1', 'x2'],
        y_column_labels=['y1']
    )

    # 데이터셋 객체 생성 (ex : tensor([[-10., 100.], ...], device = cpu), tensor([[327.7900], ...], device = cpu))
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
        criterion=nn.MSELoss(),
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        learning_rate=0.0001,
        check_point_file_save_directory_path="../check_point_files",
        # check_point_load_file_full_path="../check_points/checkpoint(2024_02_29_17_51_09_330).pt"
    )

    # 모델 저장
    save_file_full_path = tu.save_model_file(
        model=model,
        model_file_save_directory_path="../model_files"
    )

    # # 저장된 모델 불러오기
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)

    # 불러온 모델 테스트
    tu.validate_model(
        device=device,
        model=model,
        validation_dataloader=validation_dataloader
    )


if __name__ == '__main__':
    main()
