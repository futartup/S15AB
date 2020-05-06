import json
import argparse
import uuid
from datetime import datetime
from library.model.get_model import GetModel
from library.loader.data_loader import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import *
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from library.lr_finder import LRFinder
from torch.nn import *
import torch.nn.functional as F


class Main:
    """
    main class where the program diverge to various modules
    It is as an entry point
    """
    def __init__(self, conf, channels=3, height=32, width=32, data_dir='./data'):   
        # Sanity check 
        assert bool(conf) == True, "Please set configurations for your journey"
        assert "model" in conf, "Please define the model name"

        self.conf = conf 
        self.channels = channels
        self.height = height
        self.width = width
        self.data_dir = data_dir
        self.model = self.get_model()
        assert self.conf['loss'] in globals(), "The loss function name doesn't match with names available"
        self.criterion = globals()[self.conf['loss']]

        self.execution_flow()

    def execution_flow(self):
        self.get_loaders()
        self.lr_finder()
        self.get_optimizer()
        self.get_scheduler()
        
        for e in range(1, self.conf['epochs']):
            self.train()
            self.test()

        # Save the model, optimizer, 
        state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losslogger': losslogger, }
        torch.save(state, 'saved_models/{}_{}'.format(datetime.now(), uuid.uuid4()))

    def get_model(self):
        model_obj = GetModel(self.conf, self.channels, self.height, self.width )
        self.model, self.device = model_obj.return_model()
        return self.model 

    def get_loaders(self):
        obj = DataLoader(self.conf, self.data_dir)
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
        total = 0
        device = self.device

        for batchidx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            self.optimizer.zero_grad()
            # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
            # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = self.criterion(y_pred, target)
            # criteria = L1Loss(size_average=False)
            # regularizer_loss = 0
            # for param in model.parameters():
            #     regularizer_loss += criteria(param, torch.zeros_like(param))
            # loss += 0.0005 * regularizer_loss
            train_losses.append(loss)

            # Backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            train_loss += loss.item()
            predicted = y_pred.argmax(dim=1, keepdim=True)
            total += len(data)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            processed += len(data)

            accuracy = 100*correct/total
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batchidx} Accuracy={accuracy:0.2f}')
            train_acc.append(100*correct/processed)
            #return accuracy 
    
    def get_optimizer(self):
        optimizer = globals()[self.conf['optimizer']['type']]
        self.conf['optimizer'].pop('type')
        self.optimizer = optimizer(self.model.parameters(),
                                    lr=self.max_lr/10,
                                    **self.conf['optimizer'])

    def get_scheduler(self):
        if self.conf['scheduler']['type'] in scheduler_mapping:
            scheduler = globals()[self.conf['scheduler']['type']]
            self.conf['scheduler'].pop('type')
            self.scheduler = scheduler(self.optimizer,
                                       max_lr=self.max_lr,
                                       steps_per_epoch=int(len(self.train_loader))+1
                                       **self.conf['scheduler'])

    def lr_finder(self):
        criterion = globals()[self.conf['loss']]
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
    Main(conf, args.get('channels'), args.get('height'), args.get('width'), args.get('data_dir'))
