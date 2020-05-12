import json
import argparse
import uuid
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
        
        for e in range(1, self.conf['epochs']):
            self.train()
            self.test()

        # Save the model, optimizer, 
        state = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict(), 'losslogger': losslogger, }
        torch.save(state, 'saved_models/{}_{}'.format(datetime.now(), uuid.uuid4()))

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
        plt.show()

    def get_model(self):
        model_obj = GetModel(self.conf, self.height, self.width )
        self.model = model_obj.return_model()
        self.device = model_obj.get_device()
        #return self.model 

    def get_loaders(self):
        obj = DepthDataLoader(self.conf, 
                              self.data_dir + '/fg_bg/temp', 
                              self.data_dir + '/masked_images_blackwhite/temp',  
                              self.data_dir + '/depth/color/temp',
                              30)
        self.train_loader = obj.get_train_loader()
        self.test_loader = obj.get_test_loader()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                #loss = criterion(output, target)
                #loss = F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                result = pred.eq(target.view_as(pred))
                if e == 2:
                    for i,val in enumerate(result):
                        status = val.item()
                        if status:
                            if len(correct_predicted) < sample_count:
                                correct_predicted.append({
                                    'prediction': pred[i],
                                    'label': list(target.view_as(pred))[i],
                                    'image': data[i]
                                })
                            else:
                                if len(false_predicted) < sample_count:
                                    false_predicted.append({
                                        'prediction': pred[i],
                                        'label': list(target.view_as(pred))[i],
                                        'image': data[i]
                                    })
                correct += result.sum().item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        #pbar.set_description(desc= f'Loss={test_loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        a = 100. * correct / len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), a
            ))
        
        test_acc.append(100. * correct / len(test_loader.dataset))
        return test_loss

    def train(self):
        self.model.train()
        pbar = tqdm(self.train_loader)
        correct = 0
        processed = 0
        train_loss = 0
        train_acc = []
        total = 0
        global_step = 0
        device = self.device
        self.model.to(self.device)
        writer = SummaryWriter(comment=f'Summary')
        for batch in enumerate(pbar):
            # get samples
            images = batch[1]['image'].transpose(1, 3)
            mask = batch[1]['mask']

            #assert len(images) == len(mask)
            # assert images.shape[1] == self.model.n_channels, \
            #     f'Network has been defined with {self.model.n_channels} input channels' \
            #     f'but loaded with images having {images.shape[1]} channels. '\
            #     f'Please check the configuration file or images channels.'
            
            #images = images.transpose(1, 3)
            images = images.to(device=self.device, dtype=torch.float)
            mask_type = torch.float32 if self.model.n_classes == 1 else torch.long
            mask = mask.to(device=device, dtype=mask_type)

            mask_pred = self.model(images)
            loss = self.criterion(mask_pred, mask)
            train_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), global_step)

            pbar.set_postfix(**{'loss (batch)': loss.item()})
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        
            
            total += len(images)

            accuracy = 100*train_loss/total
            global_step += 1
            #pbar.set_description(desc= f'Loss={loss.item()} Accuracy={accuracy:0.2f}')
            train_acc.append(accuracy)
            #return accuracy 
    
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
