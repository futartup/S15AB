import torch

# Load the pretrained mobilenet_v2
def mobilenet_v2():
    return torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
