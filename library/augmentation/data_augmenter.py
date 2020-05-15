import albumentations as A 
import torchvision.transforms as T
from torchvision.transforms import *
from albumentations import *
import numpy as np
from albumentations.augmentations.transforms import *

mapping = {
    'albumentation': A,
    'pytorch': T,
    'Cutout': Cutout,
    'HorizontalFlip': HorizontalFlip,
    'Normalize': Normalize,
}

class TransfomedDataSet():
  def __init__(self, transform_dict={}):
    augment_list = self.get_augmented_list(transform_dict)
    print(augment_list)
    self.aug = A.Compose(augment_list)

  def get_augmented_list(self, transform_dict):
    augment_list = []
    if transform_dict['which'] in mapping:
        obj = transform_dict['which']
        if obj == 'albumentation':
            O = A 
            from albumentations.pytorch import ToTensor
        else:
            O = T
        if 'what' in transform_dict:
            for x in transform_dict['what']:
                name = x['name']
                x.pop('name')
                augment_list.append(globals()[name](**x))
        augment_list.append(ToTensor())
    return augment_list

  def __call__(self, image):
    # image = plt.imread(numpy.asarray(self.image_list[i][0]))
    # image = Image.fromarray(image).convert('RGB')
    # image = self.albu_aug(image=np.array(image))['image']
    # #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # return image
    np_image = np.asarray(image)
    #image = np.array(np_image)
    image = self.aug(image=np_image)
    return image
