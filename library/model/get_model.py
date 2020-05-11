import torch 
from torchsummary import summary
from library.model.resnet18 import ResNet18
from library.model.u_net import UNet

model_mapping = {
    "resnet18": ResNet18,
    "unet": UNet,
}

class GetModel:
    def __init__(self, conf, input_height=32, input_width=32):
        self.conf = conf 
        self.input_height = input_height
        self.input_width = input_width
        self.model = self.return_model()
        self.get_summary()

    def return_model(self):
        if self.conf.get('model') in model_mapping:
            model_name = self.conf.get('model').lower()
        else:
            print("The model names that you can define are resnet18, depth_prediction")
        model = model_mapping[model_name](**self.conf['model_initializer']).cuda()
        return model

    def get_device(self):
        return self.device

    def get_summary(self):
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        if use_cuda:
            torch.cuda.manual_seed(self.conf.get('seed'))
        self.device = torch.device("cuda" if use_cuda else "cpu")
        model = self.model.to(self.device)
        summary(model, input_size=(self.conf['model_initializer']['n_channels'], 
                                   self.input_height, self.input_width))
