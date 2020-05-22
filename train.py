import os
import json
import uuid
import torch
import logging
import argparse
import matplotlib
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.nn import *
from torch.optim import *
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Function
from matplotlib import pyplot as plt
from library.lr_finder import LRFinder
from library.model.get_model import GetModel
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.tensorboard import SummaryWriter
from library.loader.data_loader import DataLoader, DepthDataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


class Main:
    """
    main class where the program diverge to various modules
    It is as an entry point
    """
    def __init__(self, conf, data_dir='./data', load_model=None):   
        
        self.writer = SummaryWriter(conf['log_dir'])

        # Sanity check 
        assert bool(conf) == True, "Please set configurations for your journey"
        assert "model" in conf, "Please define the model name"

        self.conf = conf 
        self.data_dir = data_dir
        self.get_loaders() # get the test and train loaders
        self.load_model = load_model
        assert self.conf['loss'] in globals(), "The loss function name doesn't match with names available"
        self.criterion = globals()[self.conf['loss']]()
        self.criterion_depth = MSELoss()
        self.get_model()
        
        if not hasattr(self, 'optimizer'):
            self.get_optimizer() # get the optimizer
        if not hasattr(self, 'scheduler'):
            self.get_scheduler() # get the scheduler

        self.execution_flow()

    def execution_flow(self):
        
        #self.visualize_tranformed_data() # visualize the transformed data
        self.lr_finder() # Find the best lr 
        train_acc = []
        test_acc = []
        train_loss = []
        tests_loss = []

        # Globals step
        global_step_train = 0
        global_step_test = 0

        train_loss_decrease = 0
        test_loss_decrease = 0

        logging.info(f'''   Starting training:
                            Epochs:          {self.conf['epochs']}
                            Batch size:      {self.conf['batch_size']}
                            Training size:   {len(self.train_loader)}
                            Test size:       {len(self.test_loader)}
                            Device:          {self.device.type}
                            '''
                    )

        
        for e in range(1, self.conf['epochs']):
            print("================================")
            print("Epoch number : {}".format(e))
            self.train(e, train_acc, train_loss, train_loss_decrease, global_step_train)
            
            val_loss = self.test(test_acc, tests_loss, test_loss_decrease, global_step_test)
            #self.scheduler.step(val_loss)
            print("================================")

        self.plot_graphs(train_loss, tests_loss, train_acc, test_acc)
        
        # Save the current model
        current_directory = os.getcwd() 
        checkpoint = {
                        'epoch': self.conf['epochs'] + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                     }
        torch.save(checkpoint, '/content/drive/My Drive/Colab Notebooks/class-{0}_epoch_{1}_{2}_{3}.pth'.format(self.conf['model_initializer']['n_classes'], 
                                                                                                         self.conf['epochs'], 
                                                                                                         datetime.now(), 
                                                                                         uuid.uuid4()))
        self.writer.close()
        
    def plot_graphs(self, train_loss, tests_loss, train_acc, test_acc):
        plt.figure(figsize=(8,8))
        plt.plot(train_loss)
        plt.savefig("/content/drive/My Drive/Colab Notebooks/train_loss.jpg")

        plt.figure(figsize=(8,8))
        plt.plot(tests_loss)
        plt.savefig("/content/drive/My Drive/Colab Notebooks/test_loss.jpg")

        plt.figure(figsize=(8,8))
        plt.plot(train_acc)
        plt.savefig("/content/drive/My Drive/Colab Notebooks/train_acc.jpg")

        plt.figure(figsize=(8,8))
        plt.plot(test_acc)
        plt.savefig("/content/drive/My Drive/Colab Notebooks/test_acc.jpg")

    def visualize_tranformed_data(self):
        images = next(iter(self.train_loader))
        fg_bg = images['image']
        #mask = images['mask'][:5]
        #depth = images['depth'][:5]
        #final_image = fg_bg + mask + depth
        grid = torchvision.utils.make_grid(fg_bg)
        self.writer.add_image('Transformed images', grid)
        #self.writer.add_graph(self.model, images['image'])
        # count = 0
        # for im in images:
        #   im = T.ToPILImage(mode="RGB")(im)
        #   im.save('/content/drive/My Drive/Colab Notebooks/S15A-B/transformed_images/{}.jpg'.format(count))
        #   im.close()
        #   count += 1

    def get_model(self):
        model_obj = GetModel(self.conf)
        if self.load_model is None:
            self.model = model_obj.return_model()
            self.device = model_obj.get_device()
        else:
            checkpoint = torch.load(self.load_model)
            self.model = model_obj.return_model()
            self.conf['epochs'] = checkpoint.get('epoch', self.conf['epochs'])
            if 'state_dict' in checkpoint:
                self.model = self.model.load_state_dict(checkpoint.get('state_dict'))
            if 'optimizer' in checkpoint:
                self.optimizer = self.model.load_state_dict(checkpoint.get('optimizer'))
            self.device = model_obj.get_device()

    def get_loaders(self):
        obj = DepthDataLoader(self.conf, 
                              self.data_dir + '/fg_bg', 
                              self.data_dir + '/mask',  
                              self.data_dir + '/depth',
                              self.data_dir + '/bg',
                              .30)
        self.train_loader = obj.get_train_loader()
        self.test_loader = obj.get_test_loader()
        print("Total length of train and test is : {} and {}".format(len(self.train_loader), len(self.test_loader)))

    def test(self, test_acc, tests_loss, test_loss_decrease, global_step_test):
        self.model.eval()
        test_loss = 0
        #correct = 0
        pbar = tqdm(self.test_loader)
        length = len(self.test_loader)
        print("Length of test loader is {}".format(length))

        with torch.no_grad():
            for batch in pbar:
                images, mask, depth = batch['image'], batch['mask'], batch['depth']

                images = images.to(device=self.device, dtype=torch.float32)
                #mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
                mask = mask.to(device=self.device, dtype=torch.float32)
                depth = depth.to(device=self.device, dtype=torch.float32)

                mask_pred = self.model(images)
                #pred = torch.sigmoid(mask_pred)
                #pred = (pred > 0.5).float()
                #loss = dice_coeff(pred, mask)
                loss_m = self.criterion(mask_pred, mask.unsqueeze(1)) 
                loss_d = self.criterion_depth(mask_pred.view(depth.size()), depth)

                test_loss += loss_m.item() + loss_d.item()
                tests_loss.append(test_loss)
                
                self.writer.add_scalar('Loss/test', test_loss, global_step_test)

                accuracy = 100 * (test_loss/length)

                pbar.set_description(desc= f'Loss={test_loss} Loss={accuracy:0.2f}')
                test_acc.append(test_loss)
                global_step_test += 1
        self.model.train()
        return test_loss
  
    def train(self, epoch, train_acc, train_los, train_loss_decrease, global_step_train):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss = 0
        #train_acc = []
        length = len(self.train_loader)
        print("Length of train loader is {}".format(length))
        device = self.device
        self.model.to(self.device)
        for batch in pbar:
            # get samples
            images = batch['image'] # fg_bg images
            mask = batch['mask'] # the mask images
            depth = batch['depth'] # the depth images produced from densedepth

            images = images.to(device=self.device, dtype=torch.float)
            #mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
            mask = mask.to(device=device, dtype=torch.float32)
            depth = depth.to(device=device, dtype=torch.float32)

            mask_pred = self.model(images)
            loss_m = self.criterion(mask_pred, mask.unsqueeze(1)) 
            loss_d = self.criterion_depth(mask_pred.view(depth.size()), depth)
            final_loss = loss_m + loss_d
            train_los.append(final_loss)
            #loss_depth = self.criterion(mask_pred, depth)
            #loss = loss_mask 

            train_loss_decrease += loss_m.item() + loss_d.item()
            
            self.writer.add_scalar('Loss/train', final_loss, global_step_train)
            self.writer.add_scalar('LR/train', torch.tensor(self.scheduler.get_last_lr()), global_step_train)
            
            self.optimizer.zero_grad()
            
            final_loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            accuracy = 100*(train_loss_decrease/length)
            pbar.set_description(desc= f'Loss={final_loss.item()} Loss ={accuracy:0.2f}')
            train_acc.append(accuracy)
            self.writer.add_images('masks/images', images, global_step_train)
            self.writer.add_images('masks/true', mask.unsqueeze(1), global_step_train)
            self.writer.add_images('masks/pred', torch.sigmoid(mask_pred) > 0.5, global_step_train)
            self.writer.add_images('masks/depth', depth.unsqueeze(1), global_step_train)
            global_step_train += 1   
                     
    
    def get_optimizer(self):
        optimizer = globals()[self.conf['optimizer']['type']]
        self.conf['optimizer'].pop('type')
        if not hasattr(self, 'max_lr'):
            self.max_lr = 0.04570881896148755

        self.optimizer = optimizer(self.model.parameters(),
                                    lr=self.max_lr,
                                    **self.conf['optimizer'])

    def get_scheduler(self):
        scheduler = globals()[self.conf['scheduler']['type']]
        params = {}
        if 'OneCycleLR' == self.conf['scheduler']['type']:
            params['epochs'] = self.conf['epochs']
            params['optimizer'] = self.optimizer
            params['steps_per_epoch'] = len(self.train_loader)
            params['max_lr'] = self.max_lr
        elif 'ReduceLROnPlateau' == self.conf['scheduler']['type']:
            params['optimizer'] = self.optimizer
            params['mode'] = 'min'
        self.conf['scheduler'].pop('type')
        #if params.keys() in self.conf['scheduler'].keys():
        #    raise Exception("Duplicate keys found in conf file. Please check the readme file in github")
        self.scheduler = scheduler(**params, **self.conf['scheduler'])

    def lr_finder(self):
        criterion = globals()[self.conf['loss']]()
        optimizer = globals()["SGD"](self.model.parameters(), **self.conf['lr_finder']['optimizer'])
        lr_finder = LRFinder(self.model, optimizer, criterion, self.device) #implemented LRFinder for SGD
        lr_finder.range_test(self.train_loader, num_iter=len(self.train_loader)*10, **self.conf['lr_finder']['range_test'])
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset()
        loss = lr_finder.history['loss']
        lr = lr_finder.history['lr']
        max_lr = lr[loss.index(min(loss))]
        print("The max lr found as : {}".format(max_lr))
        self.max_lr = max_lr 


if __name__ == '__main__':
    # Main file
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf_dir", required=True,
                    help="path to the configuration file")
    ap.add_argument("--channels", required=False,
                    help="The number of channels in an image")
    ap.add_argument("--height", required=False,
                    help="The height of an image")
    ap.add_argument("--width", required=False,
                    help="The width of an image")
    ap.add_argument("--data_dir", required=True,
                    help="The Directory to the data")
    ap.add_argument("--load_model", required=False,
                    help="Load the saved model")
    args = vars(ap.parse_args())
    conf = args.get('conf_dir')
    with open(conf, 'r') as fp:
        conf = json.load(fp)
    Main(conf, args.get('data_dir'), args.get('load_model'))
