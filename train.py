import os
import sys
import json
import time
import copy
import uuid
import torch
import logging
import argparse
import matplotlib
import jsonschema
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.nn import *
from torch.optim import *
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib import pyplot as plt
import torch.nn.utils.prune as prune
from library.lr_finder import LRFinder
from library.custom_loss import DiceCoeff
from library.model.get_model import GetModel
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.tensorboard import SummaryWriter
from library.loader.data_loader import DataLoader, DepthDataLoader
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau


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
        self.criterion = globals()[self.conf['loss']['mask'][0]]()
        self.criterion_depth = globals()[self.conf['loss']['depth'][0]]()
        self.get_model()
        
        if not hasattr(self, 'optimizer'):
            self.get_optimizer() # get the optimizer
        if not hasattr(self, 'scheduler'):
            self.get_scheduler() # get the scheduler

        self.execution_flow()

    def execution_flow(self):
        
        #self.visualize_tranformed_data() # visualize the transformed data
        if self.conf['lr_finder_use']:
            self.lr_finder() # Find the best lr 
        
        val_acc_history = []
        global_step = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0
        for e in range(1, self.conf['epochs']):
            since = time.time()
            print("================================")
            print("Epoch number : {}".format(e))
            if 'prunning' in self.conf['model_optimization'] and self.conf['model_optimization']['prunning']:
                parameters_to_prune = (
                                        (self.model.down1, 'weight'),
                                        (self.model.down2, 'weight'),
                                        (self.model.down3, 'weight'),
                                        (self.model.down4, 'weight'),
                                        (self.model.up1, 'weight'),
                                        (self.model.up2, 'weight'),
                                        (self.model.up3, 'weight'),
                                        (self.model.up4, 'weight'),
                                        (self.model.outc, 'weight'),
                                      )
                prune.global_unstructured(
                                            parameters_to_prune,
                                            pruning_method=prune.L1Unstructured,
                                            amount=0.2,
                                         )
                self.sparsity()
            # self.train(e, train_acc, train_loss, train_loss_decrease, global_step_train)
            # val_loss = self.test(test_acc, tests_loss, test_loss_decrease, global_step_test)
            # self.scheduler.step(val_loss)
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                
                total = 0
                running_loss = 0
                running_corrects = 0

                for batch in self.dataloaders[phase]:
                    bg = batch['bg'].to(device=self.device, dtype=torch.float) # bg images
                    fg_bg = batch['image'].to(device=self.device, dtype=torch.float) # fg_bg images
                    mask = batch['mask'].to(device=self.device, dtype=torch.float) # the mask images
                    depth = batch['depth'].to(device=self.device, dtype=torch.float) # the depth images produced from densedepth

                    # make the parameter gradients zero
                    self.optimizer.zero_grad()

                    # forward
                    # track the history only when we are in training phase
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get the model inputs, which are fg_bg and bg.
                        # 
                        # Send the images to model, and get the output
                        images = torch.cat([fg_bg, bg], dim=1).to(device=self.device, dtype=torch.float)
                        mask_pred, depth_pred = self.model(images)
                        
                        mask_loss = self.criterion(mask_pred, mask.unsqueeze(1)) # the mask loss
                        depth_loss = self.criterion_depth(depth_pred, depth.unsqueeze(1)) # the depth loss
                        loss = mask_loss + 0.4 * depth_loss
                            
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                            self.scheduler.step()

                    running_loss += loss.item()
                    _, predicted_mask = torch.max(mask_pred.data, 1)
                    _, predicted_depth = torch.max(depth_pred.data, 1)
                    
                    total += mask.nelement() + depth.nelement()
                    running_corrects += predicted_mask.eq(mask.data).sum().item() + predicted_depth.eq(depth.data).sum().item()
            
                    # write to tensorboard
                    self.writer.add_images('input/images', images, global_step)
                    self.writer.add_images('masks/true', mask.unsqueeze(1), global_step)
                    self.writer.add_images('masks/pred', torch.sigmoid(mask_pred)*255, global_step)
                    self.writer.add_images('masks/depth', depth_pred.unsqueeze(1), global_step)
            
            epoch_loss = running_loss / len(self.dataloaders[phase])
            epoch_acc = running_corrects.double() / len(self.dataloaders[phase])
            
            # Load to tesnsorboard
            self.writer.add_scalar('Loss/{}'.format(phase), epoch_loss, global_step)
            self.writer.add_scalar('Loss/{}'.format(phase), epoch_acc, global_step)

            # Print to console
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                current_directory = os.getcwd() 
                checkpoint = {
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                     }
                torch.save(checkpoint, current_directory + '/saved-models/epoch_{1}_{2}_{3}.pth'.format( 
                                                                                                         self.conf['epochs'], 
                                                                                                         datetime.now(), 
                                                                                                         uuid.uuid4())
                                                                                                       )
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            global_step += 1
            print("================================")

        print('Best val Acc: {:4f}'.format(best_acc))
        self.plot_graphs(train_loss, tests_loss, train_acc, test_acc)
        
        self.writer.close()

    def sparsity(self):
        print(
                "Sparsity in self.model.down1.weight: {:.2f}%".format(
                    100. * float(torch.sum(self.model.down1.weight == 0))
                    / float(self.model.down1.weight.nelement())
                )
        )
        print(
            "Sparsity in self.model.down2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.down2.weight == 0))
                / float(self.model.down2.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.down3.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.down3.weight == 0))
                / float(self.model.down3.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.down4.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.down4.weight == 0))
                / float(self.model.down4.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up1.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up1.weight == 0))
                / float(self.model.up1.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up2.weight == 0))
                / float(self.model.up2.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up3.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up3.weight == 0))
                / float(self.model.up3.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up4.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up4.weight == 0))
                / float(self.model.up4.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.outc.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.outc.weight == 0))
                / float(self.model.outc.weight.nelement())
            )
        )
        print(
            "Global sparsity: {:.2f}%".format(
                100. * float(
                    torch.sum(self.model.down1.weight == 0)
                    + torch.sum(self.model.down2.weight == 0)
                    + torch.sum(self.model.down3.weight == 0)
                    + torch.sum(self.model.down4.weight == 0)
                    + torch.sum(self.model.up1.weight == 0)
                    + torch.sum(self.model.up2.weight == 0)
                    + torch.sum(self.model.up3.weight == 0)
                    + torch.sum(self.model.up4.weight == 0)
                    + torch.sum(self.model.outc.weight == 0)
                )
                / float(
                    self.model.down1.weight.nelement()
                    + self.model.down2.weight.nelement()
                    + self.model.down3.weight.nelement()
                    + self.model.down4.weight.nelement()
                    + self.model.up1.weight.nelement()
                    + self.model.up2.weight.nelement()
                    + self.model.up3.weight.nelement()
                    + self.model.up4.weight.nelement()
                    + self.model.outc.weight.nelement()
                )
            )
        )
        
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
        #self.train_loader = obj.get_train_loader()
        #self.test_loader = obj.get_test_loader()
        self.dataloaders = {}
        self.dataloaders['train'] = obj.get_train_loader()
        self.dataloaders['test'] = obj.get_test_loader()
        print("Total length of train and test is : {} and {}".format(len(self.dataloaders['train']), len(self.dataloaders['test'])))

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
                #mask = mask.to(device=self.device, dtype=torch.float32)
                depth = depth.to(device=self.device, dtype=torch.float32)

                mask_pred = self.model(images)
                #pred = torch.sigmoid(mask_pred)
                #pred = (pred > 0.5).float()
                #loss = dice_coeff(pred, mask)
                #loss_m = self.criterion(mask_pred, mask.unsqueeze(1)) 
                loss_d = self.criterion_depth(mask_pred.view(depth.size()), depth)

                test_loss += loss_d.item()
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
            #mask = batch['mask'] # the mask images
            depth = batch['depth'] # the depth images produced from densedepth

            images = images.to(device=self.device, dtype=torch.float)
            #mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
            #mask = mask.to(device=device, dtype=torch.float32)
            depth = depth.to(device=device, dtype=torch.float32)

            mask_pred = self.model(images)
            #loss_m = self.criterion(mask_pred, mask.unsqueeze(1)) 
            loss_d = self.criterion(mask_pred, depth.unsqueeze(1))
            #final_loss = loss_d
            train_los.append(loss_d)
            #loss_depth = self.criterion(mask_pred, depth)
            #loss = loss_mask 

            train_loss_decrease += loss_d.item()
            
            self.writer.add_scalar('Loss/train', train_loss_decrease, global_step_train)
            #self.writer.add_scalar('LR/train', torch.tensor(self.scheduler.get_last_lr()), global_step_train)
            
            self.optimizer.zero_grad()
            
            loss_d.backward()
            
            self.optimizer.step()
            #self.scheduler.step()
            
            accuracy = 100*(train_loss_decrease/length)
            pbar.set_description(desc= f'Loss={loss_d.item()} Loss ={accuracy:0.2f}')
            train_acc.append(accuracy)
            self.writer.add_images('masks/images', images, global_step_train)
            #self.writer.add_images('masks/true', mask.unsqueeze(1), global_step_train)
            self.writer.add_images('masks/pred', torch.sigmoid(mask_pred)*255, global_step_train)
            self.writer.add_images('masks/depth', depth.unsqueeze(1), global_step_train)
            global_step_train += 1   
                     
    
    def get_optimizer(self):
        optimizer = globals()[self.conf['optimizer']['type']]
        self.conf['optimizer'].pop('type')
        if not hasattr(self, 'max_lr'):
            #self.max_lr = 0.04570881896148755
            self.max_lr = self.conf['lr']

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
    BASE_DIR = os.getcwd()
    # Open the config json file
    with open(conf, 'r') as cfp:
        conf = json.load(cfp)
        # Check the json schema of config
        with open(BASE_DIR + '/config-schema.json', 'r') as fp:
            json_schema = json.load(fp)
            error_messages = []
            # v = jsonschema.Draft6Validator(json_schema)
            # errors = sorted(v.iter_errors(conf), key=lambda e: e.path)
            # for error in errors:
            #         error_messages.append({error.message: error.instance})
            # print(error_messages)
            # if bool(error_messages):
            #     sys.exit(0)
    
        Main(conf, args.get('data_dir'), args.get('load_model'))
