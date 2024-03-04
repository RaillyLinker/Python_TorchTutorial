import torch

# torch 버전 확인
print("torch 버전 확인")
print(torch.__version__)
print("")

# CUDA 가속 확인
print("CUDA 가속 확인")
print(torch.cuda.is_available())
if torch.cuda.is_available() :
    print(torch.cuda.get_device_name())
print(torch.version.cuda)
print("")

# MPS 가속 확인
print("MPS 가속 확인")
print(torch.backends.mps.is_built())
print(torch.backends.mps.is_available())
print("")
