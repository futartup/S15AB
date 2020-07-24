import torch

# Load the pretrained mobilenet_v2
def mobilenet_v2():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
    model.features[0][0] = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
    for param in model.parameters():
        param.requires_grad = False
    num_filters = model.classifier[1].in_features

    # Modified the last layer 
    model.classifier[1] = torch.nn.Linear(num_filters, 4)
    model.classifier[1].requires_grad = True
    return model 
