import torch
import pandas as pd
from torch import optim
from torch.utils.data import Dataset, random_split
from datetime import datetime
import os


# GPU 디바이스 값 반환
def get_gpu_support_device(
        # GPU 지원 설정 (True 로 설정 해도 현재 디바이스가 GPU 를 지원 하지 않으면 CPU 를 사용 합니다.)
        gpu_support
):
    if gpu_support:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"

    print(f"Device Selected : {device}")
    return device


# 모델 파일 저장
def save_model_file(
        # 파일로 저장할 모델 객체
        model,
        # 생성된 모델 파일을 저장할 폴더 위치 (ex : "../models")
        model_file_save_directory_path
):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    model_file_name = f"model({current_time}).pt"

    save_file_full_path = f"{model_file_save_directory_path}/{model_file_name}"
    torch.save(
        model,
        save_file_full_path
    )

    print(f"Model File Saved : {save_file_full_path}")
    return save_file_full_path


# CSV 파일을 읽고 데이터를 반환 합니다.
# output ex : [{x : [x1, x2], y : [y1]}, {x : [x1, x2], y : [y1]}, ...]
def get_csv_file_data(
        # CSV 파일 경로 (ex : "../_datasets/linear.csv")
        csv_file_full_url,
        # x 컬럼 라벨 리스트 (ex : ['x1', 'x2'])
        x_column_labels,
        # y 컬럼 라벨 리스트 (ex : ['y1'])
        y_column_labels
):
    # CSV 파일 읽기
    data_frame = pd.read_csv(csv_file_full_url)

    # 데이터 프레임을 리스트로 변환
    data_list = data_frame.to_dict(orient='records')

    # 데이터 가공
    formatted_data = []
    for item in data_list:
        x_values = [item[column] for column in x_column_labels]
        y_values = [item[y_column] for y_column in y_column_labels]
        formatted_data.append({'x': x_values, 'y': y_values})

    print(f"CSV File Data : {formatted_data}")
    return formatted_data


# 모델 사용 데이터 셋
class ModelDataset(Dataset):
    def __init__(
            self,
            # [{x : [x1, ...], y : [y1, ...]}, {x : [x1, ...], y : [y1, ...]}, ...]
            input_data
    ):
        self.data = [
            {
                "x": torch.FloatTensor(item["x"]),
                "y": torch.FloatTensor(item["y"])
            }
            for item in input_data
        ]
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]["x"], self.data[index]["y"]

    def __len__(self):
        return self.length


# Dataset 분리
def split_dataset(
        # 분리할 데이터 셋
        dataset,
        # 학습 데이터 비율 (ex : 0.8)
        train_data_rate,
        # 검증 데이터 비율 (ex : 0.2)
        validation_data_rate
):
    # rate 파라미터들의 합이 1인지 확인
    total_rate = train_data_rate + validation_data_rate
    assert total_rate == 1.0, f"Data split rates do not add up to 1.0 (current total: {total_rate})"

    # 전체 데이터 사이즈
    dataset_size = len(dataset)
    print(f"Total Data Size : {dataset_size}")

    # 목적에 따라 데이터 분리
    train_size = int(dataset_size * train_data_rate)  # 학습 데이터
    validation_size = int(dataset_size * validation_data_rate)  # 검증 데이터

    # 학습, 검증, 테스트 데이터를 무작위로 나누기
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")

    return train_dataset, validation_dataset


def train_model(
        # 사용할 디바이스
        device,
        # 학습할 모델
        model,
        # 손실 함수 (ex : nn.MSELoss())
        criterion,
        # 학습 데이터 로더
        train_dataloader,
        num_epochs=1000,  # 학습 에포크 수
        validation_dataloader=None,  # 검증 데이터 로더(None 이라면 검증은 하지 않음)
        learning_rate=0.0001,  # 학습률
        # 체크포인트 파일 저장 폴더 경로 - None 이라면 저장하지 않음 (ex : "../check_points")
        check_point_file_save_directory_path=None,
        # 불러올 체크포인트 파일 경로 - None 이라면 불러 오지 않음 (ex : "../check_points/checkpoint(2024_02_29_17_51_09_330).pt")
        check_point_load_file_full_path=None
):
    # 학습 시작 시간 기록
    start_time = datetime.now()

    print("Model Training Start!")
    # 모델 디바이스 설정
    model.to(device)

    # 손실 함수 디바이스 설정
    criterion.to(device)

    # 옵티마이저
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # 모델에 학습 모드 설정 (Dropout, Batchnorm 등의 기능 활성화)
    model.train()

    # 체크포인트 불러오기
    if check_point_load_file_full_path is not None:
        checkpoint = torch.load(check_point_load_file_full_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Check Point Loaded")
        print(f"ckp_description : {checkpoint['description']}")

    # 학습 루프
    for training_epoch in range(num_epochs):
        training_loss = 0.0

        # 배치 학습
        for tx, ty in train_dataloader:
            tx = tx.to(device)
            ty = ty.to(device)

            # 모델 순전파
            model_out = model.forward(tx)

            # 모델 결과물 loss 계산
            loss = criterion(model_out, ty)

            # 옵티마이저 기울기 초기화
            optimizer.zero_grad()
            # loss 에 따른 신경망 역전파 후 기울기 계산
            loss.backward()
            # 신경망 학습
            optimizer.step()

            training_loss += loss

        training_loss = training_loss / len(train_dataloader)

        # 1000 에 한번 실행
        if (training_epoch + 1) % 1000 == 0:
            # 학습 시작부터 얼마나 시간이 지났는지 계산
            elapsed_time = datetime.now() - start_time
            validation_loss_string = "None"
            validation_loss = 0
            if validation_dataloader is not None:
                # 검증 계산
                validation_loss = 0.0

                model.eval()  # Dropout, Batchnorm 등의 기능 비활성화
                with torch.no_grad():  # Gradient 추적 계산을 하지 않음
                    for vx, vy in validation_dataloader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        model_out = model.forward(vx)

                        loss = criterion(model_out, vy)
                        validation_loss += loss
                model.train()
                validation_loss = validation_loss / len(validation_dataloader)
                validation_loss_string = f"{validation_loss:.3f}"

            print(
                f"\nTrainingEpoch : {training_epoch + 1:4d},\n"
                f"TrainingLoss : {training_loss:.3f},\n"
                f"ValidationLoss : {validation_loss_string},\n"
                f"TL-VL ABS : {abs(training_loss - float(validation_loss)):.3f}\n"
                f"Elapsed Time: {elapsed_time}"
            )

            if check_point_file_save_directory_path is not None:
                if not os.path.exists(check_point_file_save_directory_path):
                    os.makedirs(check_point_file_save_directory_path)

                # 학습 체크포인트 저장
                current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                check_point_file_name = f"checkpoint({current_time}).pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "training_loss": training_loss,
                        "validation_loss": validation_loss_string,
                        "model": "CustomModel",
                        "description": f"CustomModel 체크포인트-{current_time}"
                    },
                    f"{check_point_file_save_directory_path}/{check_point_file_name}",
                )
                print(f"CheckPoint File Saved : {check_point_file_name}")

    print("\nModel Training Complete!")
