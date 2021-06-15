import torch


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    print(f'Device used: {device}')
    return device
