import torch
import pandas as pd
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime


# 신경망 모델 커스터마이징
class CustomModel(nn.Module):
    def __init__(
            self,
            # GPU 지원 설정 (True 로 설정 해도 현재 디바이스가 GPU 를 지원 하지 않으면 CPU 를 사용 합니다.)
            gpu_support=True
    ):
        super().__init__()
        # 모델 내 레이어
        self.linearLayer = nn.Linear(2, 1)

        # 모델 디바이스 설정
        self.device = "cpu"
        self.set_model_gpu_support(gpu_support=gpu_support)

    """
    description : 선형 회귀 모델
    input_shape : [[x1, x2], [x1, x2], [x1, x2], ...]
    output_shape : [[y1], [y1], [y1], ...]
    """

    def forward(self, model_in):
        model_out = self.linearLayer(model_in)
        return model_out

    # GPU 지원 설정
    def set_model_gpu_support(
            self,
            # GPU 지원 설정 (True 로 설정 해도 현재 디바이스가 GPU 를 지원 하지 않으면 CPU 를 사용 합니다.)
            gpu_support
    ):
        if gpu_support:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = "cpu"

        self.to(self.device)

        print(f"Model Device Set : {self.device}")

    def train_model(
            self,
            train_dataloader,  # 학습 데이터 로더
            num_epochs=1000,  # 학습 에포크 수
            validation_dataloader=None,  # 검증 데이터 로더(None 이라면 검증은 하지 않음)
            learning_rate=0.0001,  # 학습률
            check_point_file_save_directory_path=None,  # 체크포인트 파일 저장 폴더 경로(None 이라면 저장하지 않음)
            check_point_load_file_full_path=None  # 불러올 체크포인트 파일 경로(None 이라면 불러 오지 않음)
    ):
        print("Model Training Start!")

        # 손실 함수
        criterion = nn.MSELoss().to(self.device)

        # 옵티마이저
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        # Dropout, Batchnorm 등의 기능 활성화
        self.train()

        # 체크포인트 불러오기
        if check_point_load_file_full_path is not None:
            checkpoint = torch.load(check_point_load_file_full_path)
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Check Point Loaded")
            print(f"ckp_description : {checkpoint['description']}")

        # 학습 루프
        for training_epoch in range(num_epochs):
            training_loss = 0.0

            # 배치 학습
            for tx, ty in train_dataloader:
                tx = tx
                ty = ty

                # 모델 순전파
                model_out = self.forward(tx)

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
                validation_loss_string = "None"
                validation_loss = 0
                if validation_dataloader is not None:
                    # 검증 계산
                    validation_loss = 0.0

                    self.eval()  # Dropout, Batchnorm 등의 기능 비활성화
                    with torch.no_grad():  # Gradient 추적 계산을 하지 않음
                        for vx, vy in validation_dataloader:
                            vx = vx
                            vy = vy

                            model_out = self.forward(vx)

                            loss = criterion(model_out, vy)
                            validation_loss += loss
                    self.train()
                    validation_loss = validation_loss / len(validation_dataloader)
                    validation_loss_string = f"{validation_loss:.3f}"

                print(
                    f"\nTrainingEpoch : {training_epoch + 1:4d},\n"
                    f"TrainingLoss : {training_loss:.3f},\n"
                    f"ValidationLoss : {validation_loss_string},\n"
                    f"TL-VL ABS : {abs(training_loss - float(validation_loss)):.3f}"
                )

                if check_point_file_save_directory_path is not None:
                    # 학습 체크포인트 저장
                    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
                    check_point_file_name = f"checkpoint({current_time}).pt"
                    torch.save(
                        {
                            "model_state_dict": self.state_dict(),
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

    # 모델 파일 저장
    def save_model_file(
            self,
            model_file_save_directory_path):
        current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
        model_file_name = f"model({current_time}).pt"

        save_file_full_path = f"{model_file_save_directory_path}/{model_file_name}"
        torch.save(
            self,
            save_file_full_path
        )

        print(f"Model File Saved : {save_file_full_path}")
        return save_file_full_path

    # 본 모델에서 사용 하는 데이터 셋
    class ModelDataset(Dataset):
        def __init__(
                self,
                device,
                # [{x : [x1, x2], y : [y1]}, {x : [x1, x2], y : [y1]}, ...]
                input_data
        ):
            self.data = [{"x": torch.FloatTensor(item["x"]).to(device), "y": torch.FloatTensor(item["y"]).to(device)}
                         for item in input_data]
            self.length = len(self.data)

        def __getitem__(self, index):
            return self.data[index]["x"], self.data[index]["y"]

        def __len__(self):
            return self.length


def main():
    # 모델 생성
    model = CustomModel(gpu_support=True)

    # csv 파일에서 정보를 pandas 로 읽어오기
    data_frame = pd.read_csv("../datasets/non_linear.csv")
    input_data = []
    for index, row in data_frame.iterrows():
        input_data.append({'x': [row['x'] ** 2, row['x']], 'y': [row['y']]})
    dataset = CustomModel.ModelDataset(device=model.device, input_data=input_data)

    # 전체 데이터 사이즈
    dataset_size = len(dataset)
    print(f"Total Data Size : {dataset_size}")

    # 목적에 따라 데이터 분리
    train_size = int(dataset_size * 0.8)  # 학습 데이터 (80%)
    validation_size = int(dataset_size * 0.1)  # 검증 데이터 (10%)
    test_size = dataset_size - train_size - validation_size  # 테스트 데이터 (10%)

    # 학습, 검증, 테스트 데이터를 무작위로 나누기
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])
    print(f"Training Data Size : {len(train_dataset)}")
    print(f"Validation Data Size : {len(validation_dataset)}")
    print(f"Testing Data Size : {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

    # 모델 학습
    model.train_model(
        train_dataloader=train_dataloader,
        num_epochs=10000,
        validation_dataloader=validation_dataloader,
        learning_rate=0.0001,
        check_point_file_save_directory_path="../check_points",
        # check_point_load_file_full_path="../check_points/checkpoint(2024_02_29_17_51_09_330).pt"
    )

    # 모델 저장
    save_file_full_path = model.save_model_file(
        model_file_save_directory_path="../models"
    )

    # # 저장된 모델 불러오기
    device = model.device
    model = torch.load(save_file_full_path, map_location=device)
    print("Model Load Complete!")
    print(model)

    # 불러온 모델 테스트
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
