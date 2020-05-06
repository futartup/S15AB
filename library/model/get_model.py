import torch 
from torchsummary import summary
from .model.resnet18 import ResNet18

model_mapping = {
    "resnet18": ResNet18,
}

class GetModel:
    def __init__(self, conf, channels=3, input_height=32, input_width=32):
        self.conf = conf 
        self.channels = channels
        self.input_height = input_height
        self.input_width = input_width
        self.model = self.return_model(self.conf)

    @classmethod
    def return_model(self):
        if self.conf.get('model') in model_mapping:
            model_name = self.conf.get('model').lower()
        else:
            print("The model names that you can define are resnet18, resnet50")
        self.get_summary()
        return model_mapping[model_name], self.device

    def get_summary(self):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.manual_seed(self.conf.get('seed'))
            self.device = torch.device("cuda" if use_cuda else "cpu")
            model = net.to(self.device)
            summary(model, input_size=(self.channels, self.input_height, self.input_width))