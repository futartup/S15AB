from library.augmentation.data_augmenter import TransfomedDataSet
import torchvision


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
    if self.conf.data.lower() in data_dict:
      return data_dict[self.conf.data.lower()](root=self.path_to_save_data,
                                       train=True,
                                       download=True,
                                       transform=self.train_loader)
    else:
      return torchvision.datasets.ImageFolder(root=self.data_dir+ '/train', 
                                              transform=self.train_loader)
  
  def get_test_set(self): 
    if self.conf.data.lower() in data_dict:
      return data_dict[self.conf.data.lower()](root=self.path_to_save_data,
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
    
  