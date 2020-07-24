import os
import torch
import torchvision
import numpy as np 
from random import *
from PIL import Image
from glob import glob
from os import listdir
from imutils import paths
from os.path import splitext
from torch.utils.data import Dataset, DataLoader, random_split
from library.augmentation.data_augmenter import TransfomedDataSet


data_dict = {
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}


class DepthDataLoader:
  def __init__(self, conf, flying_birds_d, large_quadcopters_d, small_quadcopters_d, winged_drones_d,  test_data_percentage):
    self.conf = conf  

    # We will make the dataset in here
    flying_birds = os.listdir(flying_birds_d)
    large_quadcopters = os.listdir(large_quadcopters_d)
    small_quadcopters = os.listdir(small_quadcopters_d)
    winged_drones = os.listdir(winged_drones_d)

    # Store all the results into final_data
    flying_birds = [{'class': 0.1, 'image': flying_birds_d + '/' + x} for x in flying_birds if '.svg' not in x] 
    large_quadcopters = [{'class': 0.2, 'image': large_quadcopters_d + '/' + x} for x in large_quadcopters if '.svg' not in x] 
    small_quadcopters = [{'class': 0.3, 'image': small_quadcopters_d + '/' + x} for x in small_quadcopters if '.svg' not in x] 
    winged_drones = [{'class': 0.4, 'image': winged_drones_d + '/' + x} for x in winged_drones if '.svg' not in x]
    final_list = flying_birds + large_quadcopters + small_quadcopters + winged_drones
    shuffle(final_list)

    l = len(final_list)
    x = int(l * test_data_percentage)
    train_data, test_data = final_list[0:x], final_list[x+1:l-1]

    print(train_data, test_data)
    

    self.train = DepthDataSet(conf, train_data,
                                    transform=TransfomedDataSet(self.conf['transformations']['train']))
    self.test = DepthDataSet(conf, test_data,
                                    transform=TransfomedDataSet(self.conf['transformations']['test']))
    print(len(self.train))
    print(len(self.test))
  def get_train_loader(self):
    return torch.utils.data.DataLoader(self.train, 
                                       batch_size=self.conf.get('batch_size', 64),
                                       shuffle=self.conf.get('shuffle', True), 
                                       num_workers=self.conf.get('num_workers', 2),
                                       pin_memory=self.conf.get('pin_memory', True)
                                       )
    
  def get_test_loader(self):
    return torch.utils.data.DataLoader(self.test, 
                                       batch_size=self.conf.get('batch_size', 64),
                                       shuffle=self.conf.get('shuffle', True), 
                                       num_workers=self.conf.get('num_workers', 2),
                                       pin_memory=self.conf.get('pin_memory', True)
                                       )



class DepthDataSet(Dataset):
  """ Dataset for Depth and mask prediction """

  def __init__(self, conf, data, transform=None, scale=1):
    """
    Args:
        conf = configuration file
        image_dir: Directory to images of depth
        mask_dir: Directory to mask images of depth
        transformation: transformations applied on that image
    """
    self.conf = conf
    #self.fg_bg_dir = fg_bg_dir
    #self.mask_dir = mask_dir 
    #self.depth_dir = depth_dir 
    
    self.scale = scale
    self.transform = transform
    self.ids = data
    #self.bg_images = list(paths.list_images(bg_dir))

  def __len__(self):
    return len(self.ids)

  @classmethod
  def preprocess(cls, pil_img, scale):
      w, h = pil_img.size
      newW, newH = int(scale * w), int(scale * h)
      assert newW > 0 and newH > 0, 'Scale is too small'
      pil_img = pil_img.resize((newW, newH))

      img_nd = np.array(pil_img)

      if len(img_nd.shape) == 2:
          img_nd = np.expand_dims(img_nd, axis=2)

      # HWC to CHW
      img_trans = img_nd.transpose((2, 0, 1))
      if img_trans.max() > 1:
          img_trans = img_trans / 255

      return img_trans

  def __getitem__(self, i):
    idx = self.ids[i]
    # image_file = glob(self.image_dir + '/'+ idx )
    # mask_file = glob(self.mask_dir + '/'+ idx )
    # print(idx)
    # print(self.image_dir)
    # print(mask_file)

    # assert len(mask_file) > 1, "No mask found"
    # assert len(image_file) > 1, "No image found"
    # avoid svg images
    pil_image = Image.open(idx['image'])
    print(pil_image.mode)
    if pil_image.mode in ['RGBA', 'RGB']:
      pil_image = pil_image.convert('RGB')
    

    
      #mask = Image.open(self.mask_dir + '/'+ idx)
      #fg_bg = Image.open(self.fg_bg_dir + '/'+ idx)
      #depth = Image.open(self.depth_dir + '/'+ idx)
      #if image.size != mask.size:
      #assert image.size == mask.size
      #img = self.preprocess(image, self.scale)
      #mask = self.preprocess(mask, self.scale)
      #bg = self.preprocess(bg, self.scale)
      image =  self.transform(image=pil_image)
      
      #bg =  self.transform(image=bg)
      return {
              'image': image['image'], 
              'class': idx['class'],
            }
