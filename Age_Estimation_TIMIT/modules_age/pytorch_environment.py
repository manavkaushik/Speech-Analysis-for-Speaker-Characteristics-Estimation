import torch

# Check if CUDA is available #

def pytorch_env():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
        print('########### Running on CUDA ############')  
    else:
        device = torch.device("cpu")
        print('########### Running on CPU ############')  

    return device