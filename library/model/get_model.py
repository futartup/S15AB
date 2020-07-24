import torch 
from torchsummary import summary
from library.model.resnet18 import ResNet18
from library.model.u_net import UNet
from library.model.mobilenet_v2 import mobilenet_v2

model_mapping = {
    "resnet18": ResNet18,
    "unet": UNet,
    "mobilenet_v2": mobilenet_v2,
}

class GetModel:
    def __init__(self, conf):
        self.conf = conf 

    def return_model(self):
        if self.conf.get('model') in model_mapping:
            model_name = self.conf.get('model').lower()
        else:
            print("The model names that you can define are resnet18, depth_prediction")
        self.model = model_mapping[model_name](**self.conf['model_initializer'])
        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        if use_cuda:
            torch.cuda.manual_seed(self.conf.get('seed'))
        self.device = torch.device("cuda" if use_cuda else "cpu")
        model = self.model.to(device=self.device, dtype=torch.float)
        summary(model, input_size=(1, 224, 224))
        return model
        
    def get_device(self):
        return self.device
        
