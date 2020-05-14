from library.augmentation.data_augmenter import TransfomedDataSet
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from os.path import splitext
from os import listdir
from glob import glob
import numpy as np 
from PIL import Image


data_dict = {
    'cifar10': torchvision.datasets.CIFAR10,
    'cifar100': torchvision.datasets.CIFAR100,
}


class DataLoader():
  def __init__(self, conf=None, data_dir='./data', path_to_save_data='./data'):
    self.conf = conf
    self.data_dir = data_dir
    self.path_to_save_data = path_to_save_data
    self.path = path_to_save_data + '/tiny-imagenet-200'
    self.train_loader = self.get_transformed_data(True)
    self.test_loader = self.get_transformed_data(False)

  def get_transformed_data(self, train=True):
    if train:
        return TransfomedDataSet(self.conf['transformations']['train'])
    else:
        return TransfomedDataSet(self.conf['transformations']['test'])

  def download_images(self, url):
    if os.path.isdir("tiny-imagenet-200.zip"):
      print("Images are already there")
      return
    r = requests.get(url, stream=True)
    print("Downloading " + url)
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    zip_ref.extractall(self.path_to_save_data)
    zip_ref.close()
    return

  def get_train_set(self): 
    if self.conf['data'].lower() in data_dict:
      return data_dict[self.conf['data'].lower()](root=self.path_to_save_data,
                                       train=True,
                                       download=True,
                                       transform=self.train_loader)
    else:
      return torchvision.datasets.ImageFolder(root=self.data_dir+ '/train', 
                                              transform=self.train_loader)
  
  def get_test_set(self): 
    if self.conf['data'].lower() in data_dict:
      return data_dict[self.conf['data'].lower()](root=self.path_to_save_data,
                                       train=False,
                                       transform=self.test_loader)
    else:
      return torchvision.datasets.ImageFolder(root=self.data_dir + '/test', 
                                              transform=self.train_loader)
    
  def get_train_loader(self):
    return torch.utils.data.DataLoader(self.get_train_set(), 
                                       batch_size=self.conf.get('batch_size', 64),
                                       shuffle=self.conf.get('shuffle', True), 
                                       num_workers=self.conf.get('num_workers', 2))
    
  def get_test_loader(self):
    return torch.utils.data.DataLoader(self.get_test_set(), 
                                       batch_size=self.conf.get('batch_size', 64),
                                       shuffle=self.conf.get('shuffle', True), 
                                       num_workers=self.conf.get('num_workers', 2))
    
  def get_test_train_loaders(self):
     train_data, train_labels, test_data, test_labels = self.get_data(self.get_id_dictionary())
     traindata = torchvision.datasets.ImageFolder(root=self.path_to_save_data + '/tiny-imagenet-200/train', 
                                              transform=self.train_loader)
    
  

class DepthDataLoader:
  def __init__(self, conf, fg_bg_dir, mask_dir, depth_dir, test_data_percentage):
    # self.image_dir = image_dir
    # self.mask_dir = mask_dir
    self.conf = conf
    # self.test_data_percentage = test_data_percentage
    #dataset = DepthDataSet(conf, fg_bg_dir, mask_dir, depth_dir)
    #test_p = int(len(dataset) * test_data_percentage)
    #train_p = len(dataset) - test_p
    #self.train, self.test = random_split(dataset, [train_p, test_p])
    self.train = DepthDataSet(conf, fg_bg_dir+ '/train', mask_dir+'/train', depth_dir+'/train', transform=TransfomedDataSet(self.conf['transformations']['train']))
    self.test = DepthDataSet(conf, fg_bg_dir+ '/test', mask_dir+'/test', depth_dir+'/test', transform=TransfomedDataSet(self.conf['transformations']['test']))

  def get_train_loader(self):
    return torch.utils.data.DataLoader(self.train, 
                                       batch_size=self.conf.get('batch_size', 64),
                                       shuffle=self.conf.get('shuffle', True), 
                                       num_workers=self.conf.get('num_workers', 2),
                                       pin_memory=self.conf.get('pin_memory', True))
    
  def get_test_loader(self):
    return torch.utils.data.DataLoader(self.test, 
                                       batch_size=self.conf.get('batch_size', 64),
                                       shuffle=self.conf.get('shuffle', True), 
                                       num_workers=self.conf.get('num_workers', 2),
                                       pin_memory=self.conf.get('pin_memory', True))



class DepthDataSet(Dataset):
  """ Dataset for Depth and mask prediction """

  def __init__(self, conf, fg_bg_dir, mask_dir, depth_dir, transform=None, scale=1):
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

    mask = Image.open(self.mask_dir + '/'+ idx)
    fg_bg = Image.open(self.fg_bg_dir + '/'+ idx)
    depth = Image.open(self.depth_dir + '/'+ idx)
    #if image.size != mask.size:
    #assert image.size == mask.size
    #img = self.preprocess(image, self.scale)
    #mask = self.preprocess(mask, self.scale)

    return {
            'image': self.transform(image=fg_bg), 
            'mask': torch.from_numpy(np.array(mask)), 
            'depth': torch.from_numpy(np.array(depth))
           }
