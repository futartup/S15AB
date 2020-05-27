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
  def __init__(self, conf, fg_bg_dir, mask_dir, depth_dir, bg_dir,  test_data_percentage):
    self.conf = conf  
    self.train = DepthDataSet(conf, fg_bg_dir+ '/train', 
                                    mask_dir+'/train', 
                                    depth_dir+'/train', 
                                    bg_dir,
                                    transform=TransfomedDataSet(self.conf['transformations']['train']))
    self.test = DepthDataSet(conf, fg_bg_dir+ '/test', 
                                    mask_dir+'/test', 
                                    depth_dir+'/test', 
                                    bg_dir,
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

  def __init__(self, conf, fg_bg_dir, mask_dir, depth_dir, bg_dir, transform=None, scale=1):
    """
    Args:
        conf = configuration file
        image_dir: Directory to images of depth
        mask_dir: Directory to mask images of depth
        transformation: transformations applied on that image
    """
    self.conf = conf
    self.fg_bg_dir = fg_bg_dir
    self.mask_dir = mask_dir 
    self.depth_dir = depth_dir 
    
    self.scale = scale
    self.transform = transform
    self.ids = [file for file in listdir(fg_bg_dir) if not file.startswith('.')]
    self.bg_images = list(paths.list_images(bg_dir))

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
    bg = Image.open(self.bg_images[randint(1, 100)])
    if bg.mode == 'RGBA':
      print("RGBA image found converting to RBG")
      bg = bg.convert('RGB')
      print("After converting mode is {}".format(bg.mode))
    mask = Image.open(self.mask_dir + '/'+ idx)
    fg_bg = Image.open(self.fg_bg_dir + '/'+ idx)
    depth = Image.open(self.depth_dir + '/'+ idx)
    #if image.size != mask.size:
    #assert image.size == mask.size
    #img = self.preprocess(image, self.scale)
    #mask = self.preprocess(mask, self.scale)
    bg = self.preprocess(depth, self.scale)
    fg_bg =  self.transform(image=fg_bg)
    #bg =  self.transform(image=bg)
    return {
            'bg': bg,
            'image': fg_bg['image'], 
            'mask': torch.from_numpy(np.array(mask)/255), 
            'depth': torch.from_numpy(np.array(depth)/255)
           }
