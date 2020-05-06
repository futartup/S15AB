import pytorch
import albumentation as A 
import torchvision.transforms as T
from torchvision.transforms import *
from albumentation import *


class TransfomedDataSet():
  def __init__(self, transform_dict={}):
    if len(transform_dict) == 0:
        self.aug = None
    else:
        self.aug = self.get_augmented_images(transform_dict)

  def get_augmented_images(transform_list):
    augment_list = []
    if transform_list['which'] in mapping:
        obj = transform_list['which']
        if obj == 'albumentation':
            O = A 
            from albumentations.pytorch import ToTensor
        else:
            O = T
        methods = dir(obj)
        for x in transform_list['what']:
            if x['name'] in methods:
                name = x['name']
                x.pop('name')
                augment_list.append(globals()[name](**x))
        augment_list.append(ToTensor())
    return O.compose(augment_list) 

  def __call__(self, image):
    # image = plt.imread(numpy.asarray(self.image_list[i][0]))
    # image = Image.fromarray(image).convert('RGB')
    # image = self.albu_aug(image=np.array(image))['image']
    # #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # return image
    image = np.array(image)
    image = self.albu_aug(image=image)['image']
    return image
