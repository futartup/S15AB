import albumentations as A 
import torchvision.transforms as T
from torchvision.transforms import *
from albumentations import *

mapping = {
    'albumentation': A,
    'pytorch': T,
}

class TransfomedDataSet():
  def __init__(self, transform_dict={}):
    if len(transform_dict) == 0:
        self.aug = None
    else:
        self.aug = self.get_augmented_images(transform_dict)

  def get_augmented_images(self, transform_dict):
    augment_list = []
    if transform_dict['which'] in mapping:
        obj = transform_dict['which']
        if obj == 'albumentation':
            O = A 
            from albumentations.pytorch import ToTensor
        else:
            O = T
        methods = dir(obj)
        if 'what' in transform_dict:
            for x in transform_dict['what']:
                if x['name'] in methods:
                    name = x['name']
                    x.pop('name')
                    augment_list.append(globals()[name](**x))
        augment_list.append(ToTensor())
    return O.Compose(augment_list) 

  def __call__(self, image):
    # image = plt.imread(numpy.asarray(self.image_list[i][0]))
    # image = Image.fromarray(image).convert('RGB')
    # image = self.albu_aug(image=np.array(image))['image']
    # #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # return image
    image = np.array(image)
    image = self.albu_aug(image=image)['image']
    return image
