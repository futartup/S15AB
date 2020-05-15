import json
import argparse
import uuid
import logging
from datetime import datetime
from library.model.get_model import GetModel
from library.loader.data_loader import DataLoader, DepthDataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from library.lr_finder import LRFinder
from torch.nn import *
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class Main:
    """
    main class where the program diverge to various modules
    It is as an entry point
    """
    def __init__(self, conf, height=32, width=32, data_dir='./data'):   
        # Sanity check 
        assert bool(conf) == True, "Please set configurations for your journey"
        assert "model" in conf, "Please define the model name"

        self.conf = conf 
        self.height = int(height)
        self.width = int(width)
        self.data_dir = data_dir
        self.get_model()
        assert self.conf['loss'] in globals(), "The loss function name doesn't match with names available"
        self.criterion = globals()[self.conf['loss']]()

        self.execution_flow()

    def execution_flow(self):
        self.get_loaders() # get the test and train loaders
        #self.visualize_tranformed_data() # visualize the images for training
        #self.lr_finder() # find the best LR
        self.get_optimizer() # get the optimizer
        self.get_scheduler() # get the scheduler
        
        train_acc = []
        test_acc = []

        logging.info(f'''   Starting training:
                            Epochs:          {self.conf['epochs']}
                            Batch size:      {self.conf['batch_size']}
                            Training size:   {len(self.train_loader)}
                            Test size:       {len(self.test_loader)}
                            Device:          {self.device.type}
                            ''')
        writer = SummaryWriter(comment=f'BS_{self.conf['batch_size']}')
        for e in range(1, self.conf['epochs']):
            print("================================")
            print("Epoch number : {}".format(e))
            self.train(e, train_acc)
            self.test(test_acc)
            print("================================")

        self.plot_graphs(train_acc, test_acc)
        # plot the train and test accuracy graph

        # Save the model, optimizer, 
        # state = {'state_dict': self.model.state_dict(),
        #          'optimizer': self.optimizer.state_dict()}
        torch.save(self.model.state_dict(), 'saved_models/no_depth_epoch_{}_{}_{}.pth'.format(self.conf['epochs'], datetime.now(), uuid.uuid4()))

    def plot_graphs(self, train_acc, test_acc):
        plt.figure(figsize=(8,8))
        plt.plot(train_acc)
        plt.savefig("train_loss.jpg")

        plt.figure(figsize=(8,8))
        plt.plot(test_acc)
        plt.savefig("test_loss.jpg")

    def visualize_tranformed_data(self):
        images = next(iter(self.train_loader))
        images = images['image'].numpy()  # convert images to numpy for display
        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(10, 10))
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                        nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                        axes_pad=0.1,  # pad between axes in inch.
                        )
        for ax, im in zip(grid, images):
            # Iterating over the grid returns the Axes.
            ax.imshow(im)
        plt.savefig("transformed_images.png")

    def get_model(self):
        model_obj = GetModel(self.conf, self.height, self.width )
        self.model = model_obj.return_model()
        self.device = model_obj.get_device()
        #return self.model 

    def get_loaders(self):
        obj = DepthDataLoader(self.conf, 
                              self.data_dir + '/fg_bg/masked_images', 
                              self.data_dir + '/masked_images_blackwhite/temp',  
                              self.data_dir + '/depth/temp1',
                              .30)
        self.train_loader = obj.get_train_loader()
        self.test_loader = obj.get_test_loader()
        print("Total length of train and test is : {} and {}".format(len(self.train_loader), len(self.test_loader)))

    def test(self, test_acc):
        self.model.eval()
        test_loss = 0
        #correct = 0
        pbar = tqdm(self.test_loader)
        length = len(self.test_loader)
        print("Length of test loader is {}".format(length))
        test_accuracy = []
        with torch.no_grad():
            for batch in pbar:
                images, mask, depth = batch['image'].transpose(1,3), batch['mask'], batch['depth']

                images = images.to(device=self.device, dtype=torch.float32)
                mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
                mask = mask.to(device=self.device, dtype=mask_type)
                depth = depth.to(device=self.device, dtype=mask_type)

                mask_pred = self.model(images)
                loss_mask = self.criterion(mask_pred, mask)
                #loss_depth = self.criterion(mask_pred, depth)
                loss = loss_mask 

                test_loss += loss_mask.item() 

                accuracy = 100 * (test_loss/length)

                pbar.set_description(desc= f'Loss={test_loss} Accuracy={accuracy:0.2f}')
                test_acc.append(test_loss)

    def train(self, epoch, train_acc):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss = 0
        #train_acc = []
        length = len(self.train_loader)
        print("Length of train loader is {}".format(length))
        device = self.device
        self.model.to(self.device)
        for batch in enumerate(pbar):
            # get samples
            images = batch[1]['image']
            mask = batch[1]['mask']
            depth = batch[1]['depth']
            print(len(images), len(mask), len(depth))
            images = images.to(device=self.device, dtype=torch.float)
            mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
            mask = mask.to(device=device, dtype=mask_type)
            depth = depth.to(device=device, dtype=mask_type)

            mask_pred = self.model(images)
           
            loss_mask = self.criterion(mask_pred, mask)
            #loss_depth = self.criterion(mask_pred, depth)
            loss = loss_mask 

            train_loss += loss_mask.item() 
            #writer.add_scalar('Loss/train', train_loss, global_step)

            pbar.set_postfix(**{'loss (batch)': train_loss})
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            accuracy = 100*(train_loss/length)
            pbar.set_description(desc= f'Loss={train_loss} Accuracy={accuracy:0.2f}')
            train_acc.append(accuracy)            
    
    def get_optimizer(self):
        optimizer = globals()[self.conf['optimizer']['type']]
        self.conf['optimizer'].pop('type')
        try:
          self.max_lr = self.max_lr/10
        except:
          self.max_lr = 0.01
        self.optimizer = optimizer(self.model.parameters(),
                                    lr=self.max_lr/10,
                                    **self.conf['optimizer'])

    def get_scheduler(self):
        scheduler = globals()[self.conf['scheduler']['type']]
        self.conf['scheduler'].pop('type')
        self.scheduler = scheduler(self.optimizer,
                                    max_lr=self.max_lr,
                                    epochs=self.conf['epochs'],
                                    steps_per_epoch=int(len(self.train_loader))+1,
                                    **self.conf['scheduler'])

    def lr_finder(self):
        criterion = globals()[self.conf['loss']]()
        optimizer = globals()[self.conf['optimizer']['type']](self.model.parameters(), **self.conf['lr_finder']['optimizer'])
        lr_finder = LRFinder(self.model, optimizer, criterion, self.device) #implemented LRFinder for SGD
        lr_finder.range_test(self.train_loader, num_iter=len(self.train_loader)*10, **self.conf['lr_finder']['range_test'])
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset()
        loss = lr_finder.history['loss']
        lr = lr_finder.history['lr']
        max_lr = lr[loss.index(min(loss))]
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
    Main(conf, args.get('height'), args.get('width'), args.get('data_dir'))
