import torch
import numpy as np

# 텐서 생성
print("텐서 생성")
print(torch.tensor([1, 2, 3]))
print(torch.Tensor([[1, 2, 3], [4, 5, 6]]))
print(torch.LongTensor([1, 2, 3]))
print(torch.FloatTensor([1, 2, 3]))
print("")

# 텐서 속성 확인
print("텐서 속성 확인")
tensor = torch.rand(1, 2)
print(tensor)
print(tensor.shape)
print(tensor.dtype)
print(tensor.device)
print("")

# 텐서 차원 변환
print("텐서 차원 변환")
tensor = tensor.reshape(2, 1)
print(tensor)
print(tensor.shape)
print("")

# 텐서 자료형 설정
print("텐서 자료형 설정")
tensor = torch.rand((3, 3), dtype=torch.float)
print(tensor)
print("")

# 텐서 GPU 장치 설정 cuda
print("텐서 자료형 설정 cuda")
if torch.cuda.is_available():
    device = "cuda"
    tensor = torch.tensor(data=[1, 2, 3], dtype=torch.float32, device="cuda")
else:
    device = "cpu"
    tensor = torch.FloatTensor([1, 2, 3])
print(device)
print(tensor)
tensor = torch.rand((1, 1), device=device)
print(tensor)
print("")

# 텐서 GPU 장치 설정 mps
print("텐서 자료형 설정 mps")
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"
print(device)
tensor = torch.FloatTensor([1, 2, 3], device=device)
print(tensor)
print("")

# 텐서 장치 변환 cuda
if torch.cuda.is_available():
    print("텐서 장치 변환 cuda")
    cpu = torch.FloatTensor([1, 2, 3])
    print(cpu)
    gpu = cpu.cuda()
    print(gpu)
    gpu2cpu = gpu.cpu()
    print(gpu2cpu)
    cpu2gpu = cpu.to("cuda")
    print(cpu2gpu)
    print("")

# 텐서 장치 변환 mps
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("텐서 장치 변환 mps")
    cpu = torch.FloatTensor([1, 2, 3])
    print(cpu)
    gpu = cpu.to("mps")
    print(gpu)
    gpu2cpu = gpu.cpu()
    print(gpu2cpu)
    print("")

# 넘파이 배열을 텐서로
print("넘파이 배열을 텐서로")
ndarray = np.array([1, 2, 3], dtype=np.uint8)
print(torch.tensor(ndarray))
print(torch.Tensor(ndarray))
print(torch.from_numpy(ndarray))
print("")

# 텐서를 넘파이 배열로
print("텐서를 넘파이 배열로")
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = "mps"
else:
    device = "cpu"

tensor = torch.FloatTensor([1, 2, 3]).to(device)
ndarray = tensor.detach().cpu().numpy()
print(ndarray)
print(type(ndarray))
print("")
