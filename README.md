# Mask and Monocular depth Prediction ( S15A and S15 )
Initial model idea is taken from [paper implementation U-Net](https://arxiv.org/abs/1505.04597). The model is customized to take 2 inputs and 2 outputs.
This repo is to detect cars in video. The goal is to segment vehicle in images as well as monocular depth estimation. I have created my own dataset.

## Acknowledgements

I would like to thanks all the EVA4 telegram batch members who has helped me to implement the code whenever i am stuck.

I would also like to thank www.theschoolofai.in to give me this opporthunity to get hands on AI. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- GIMP

## Learnings and Challenges
### Challenges
- I am completely new into the field so the learning curve is steep.
- My planning to complete the task was completely unclear.
- My internet was fluctuating very much these days, so had to work longer.
- The keysteps identification was not clear.
- Knowledge of various models was blurry.
- Though basics was clear , but the application of these knowledges was the key.
- There were some hidden tricks and the identification of these was the challenges. I can say it as an unseen challenges.
- Google colab pro is not available in India.
- My Backpain. But im not going to make it an excuse. May the back pain.


### Learnings
- The first and foremost learning is that , couple of days ago somewhere i came to read the quote ***"The job well begun is half done"***. I realized this quote to the bone of my spine. The data management is the key to get very good results. Prepare the data with utmost energy and never compromise on that. 
- The model should be good. I tried with UNet and mix of resnet34 + UNet. There are lot of factors for which i cannot make the model very heavy as the time is fix.
- There are many corner cases from start to end. We have to take care all of these. Right from data preparation to final output there are many situations which we have to take care. For eg., as you dataset is very large , you can have many techniques through which you can send your data to google drive. 
- Patience is very important to try out many combination of models. Try from the simple one.
- Domain knowledge is very important. The environment form where the inputs will come, if we can have the knowledge of that then it will be very good.
- Data transformations are important.

### Mistakes
- The printing of the output is not very proper.
- I didn't tried from simpler models. I tried only with UNet and mix of resnet34 + UNet.
- The experimentation on resnet34 + UNet mix model is very less.

## Configuration file

[config](https://github.com/futartup/S15AB/blob/master/config.py)
```
{
    "seed": <int>,
    "lr": <float>,
    "data": <str: whether the data is build custom or inbuilt; declare custom>,
    "num_workers": <int>,
    "shuffle": <boolean>,
    "epochs": <int>,
    "multiclass": <boolean>,
    "batch_size": <int>,
    "num_workers": <int>,
    "lr_step_size": <int>,
    "lr_gamma": <float>,
    "loss": { <str: what to predict>: [<str: loss function name>], <str: what to predict another>: [<str: loss function name>]},
    "lr_finder_use": <boolean: whether lr finder to be used or not>,
    "lr_finder": {
        "optimizer": {"lr": <float>},
        "range_test": {"end_lr":<int>, "step_mode": <str>}
    },
    "optimizer": {
        "type": <str: name of the optimizer>,
        "weight_decay": <float: params goes accordingly>
    },
    "scheduler": {
        "type": <str: name of the scheduler>,
        "patience": <int: params goes accordingly>
    },
    "transformations": {
        "train": {
            "which": <str: albumentation or pytorch>,
            "what": [
            {
                "name": <str: Name of the albumentation>,
                "num_holes":<int: Params goes accordingly>
            }
        ]
        },
        "test": {
            "which": <str: albumentation or pytorch>,
            "what": <list: albumentation or pytorch transformations, got the declaration idea from train>
        }
    },
    "model": <str: name of the model>,
    "model_initializer": {
        "n_channels": <int: number of input channels>,
        "n_classes": <int: number of classes>,
        "bilinear": <boolean: whether bilinear or not>
    },
    "log_dir": <str: path to the log file, tensorboard will read it from here>
}
```
The json will be validated using jsonschema validator [here](https://github.com/futartup/S15AB/blob/master/train.py)

## To train the model
[main.ipynb](https://github.com/futartup/S15AB/blob/master/main.ipynb) can be run in 
[train.py](https://github.com/futartup/S15AB/blob/master/train.py)
```
!cd S15AB; CUDA_LAUNCH_BLOCKING=1 python3 train.py --h
usage: train.py [-h] --conf_dir CONF_DIR [--channels CHANNELS]
                [--height HEIGHT] [--width WIDTH] --data_dir DATA_DIR
                [--load_model LOAD_MODEL]

optional arguments:
  -h, --help            show this help message and exit
  --conf_dir CONF_DIR   path to the configuration file
  --channels CHANNELS   The number of channels in an image
  --height HEIGHT       The height of an image
  --width WIDTH         The width of an image
  --data_dir DATA_DIR   The Directory to the data
  --load_model LOAD_MODEL
                        Load the saved model
```

## To predict the output
[main.ipynb](https://github.com/futartup/S15AB/blob/master/main.ipynb) can be run in 
[predict1.py](https://github.com/futartup/S15AB/blob/master/predict1.py)
```
!cd S15AB; CUDA_LAUNCH_BLOCKING=1 python3 predict1.py --h
usage: predict1.py [-h] [--model FILE] --input INPUT [INPUT ...]
                   [--output INPUT [INPUT ...]] [--viz] [--no-save]
                   [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```

## Output images
| Input image             | Depth Output       | Mask Output |
|-------------------------| -------------------| -------------|
|![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/input.jpg)| ![alt-text-1(https://github.com/futartup/S15AB/blob/master/raw_images/depth.jpg) |![alt-text-1(https://github.com/futartup/S15AB/blob/master/raw_images/mask.jpg)|


## Constructing the model architecture
To solve this problem i have implemented UNet with total trainable params of 10, 043, 321.
The model code can be found [here](https://github.com/futartup/S15AB/blob/master/library/model/u_net.py)
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/model_diagram_res_unet1.jpg)

There is another implemention of mixed model of resnet34 and UNet. The encoder part is the resnet34 and the decoder part is the UNet. The model is implemented [here](https://github.com/futartup/S15AB/blob/master/library/model/res_u_net.py). The reason for choosing resnet34 is that it is light and serves most of the purpose. The resnet34 can be transfer learned. 

The saved models are stored in [here](https://drive.google.com/drive/u/3/folders/1A8jBzOUM_WPFKIa-gjjYs3H9tTffAGyL).
The code for storing the saved model can be found [here](https://github.com/futartup/S15AB/blob/master/train.py).

## Representation of task
The goal is to segment the cars(which is the case in my case) in an image and predict the depth in an image. Simply, my goal is to take either a RGB color image (height×width×3) or a grayscale image (height×width×1) and output a segmentation map where each pixel contains a class label represented as an integer (height×width×1). 

There are four kinds of inputs which are classified into bg, fg_bg, mask and depth. The explanation for these categories of images can be found below. The mask and depth are the ground truths for our images. The mask has only one class which is a vehicle class as of now only. The depth dataset has been created using this [repo](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb)

The fg_bg and bg images are feed into the model. The masks and depth are used as targets in our criterion. The loss is added up and then autograd will backprop in this total loss. But the gradients will be for these individual losses.

![alt-text-1](https://github.com/futartup/S15AB/blob/master/data/images/lady.png) 
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/depth.png)

## Loss functions
| Task             | Loss Function     |  Reason to choose |
| ----------------- | ----------- | -----------------------|
|          Mask Prediction   |  BCEWithLogitsLoss  | I want to find the loss between each pixels from fg and the mask of it. The mask is all white pixels and the surrounding is black pixels in the mask image. Therefore its a binary logits classification. |
| Depth | MSELoss | There is no class as such , i want to find the cross entropy between each input and output pixels |

Alongwith the BCEWithLogitsLoss , dice loss can also be used [code](https://github.com/futartup/S15AB/blob/master/library/custom_loss.py).
The dice coefficient can be said as the amount of overlap between target and input.
```
dice_coeff = 2 |A . B| / |A| + |B|
```
This dice loss can be use to test the accuracy of the model.

These loss functions can be defined in [here](https://github.com/futartup/S15AB/blob/master/config.py)

### Experiments which various loss functions
 Loss function             | Training loss        | 
| ----------------- | ----------- |
| Mask - BCEWithLogitsLoss, Depth- SSIM | Mask - 0.0391, Depth - 0.2233|
| Mask - BCEWithLogitsLoss, Depth - BCEWithLogitsLoss | Mask - 0.0391, Depth - 0.6537 |
| Mask - SSIM, Depth - SSIM | Mask - 0.4995, Depth - 0.2306 |
| Mask - MSELoss, Depth - MSELoss | Mask - 0.0057, Depth - 0.0312|
| Mask - L1Loss, Depth - L1Loss | Mask - 0.0514, Depth - 0.1657|
  
## Loss curves
Mask = BCEWithLogitsLoss, Depth=MSELoss, Scheduler=ReduceLROnPlateau, Optimizer=Adam

Train Loss
<img src='https://github.com/futartup/S15AB/blob/master/raw_images/train_loss%20(1).jpg' width="200" height="200">

Test Loss
<img src='https://github.com/futartup/S15AB/blob/master/raw_images/test_loss%20(1).jpg' width="200" height="200">


## Data Augmentation
***Please note that while mentioning the albumentation names it should exactly matches with the name, case with the documentation, because the name is loaded using globals() magic function. Just to prevent lot of if else condition added that. For eg., Cutout cannot be named as something else as cutout, cutOut etc. It should exactly matches with the name mentioned in albumentation or pytorch library.***
### Train dataset
#### fg_bg images
```
{
    "name": "Cutout",  (Used because i want the model to generalize on unseen data)
    "num_holes":1,
    "max_h_size": 80, 
    "max_w_size": 80,
    "always_apply": true
},
{
    "name": "Blur", (Used this because sometimes the camera can shake )
    "always_apply": false,
    "p": 0.2
},
{
    "name": "GaussNoise", (Added noise)
    "always_apply": false,
    "p": 0.2
},
{
    "name": "Normalize",
    "mean": [0.429694425216733, 0.43985525255137686, 0.43281280297561686],
    "std":[0.28715867313016924, 0.266891016687173, 0.26731143118502665],
    "always_apply": true
}
```

#### bg images
same augmentation applied to fg_bg images

#### mask and depth
```
{
    "name": "Normalize",
    "mean": [0.4316956210782655, 0.4422608771711209, 0.43416351285063187],
    "std": [0.2876632239505755, 0.26742029335370965, 0.26840916462077713],
    "always_apply": true
}
```



## Creation of dataset
### Choosing of background images
For choosing background images i have choosen the natural scenery images. The scenery images are choosen in such a way that there are very less subjects in the image. I will overlay foreground images on top of these scene images on randon positions. 
The duplicate images produces bias so these images are reduced using this [code](https://github.com/futartup/S15A/blob/master/detect_and_remove_duplicate_images.py).

The number of these backgroud images are 100 of size 224 * 224.
The scene images are as follows-

![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/background_images_224/16ms-17c-sardar-petal-road-adyar-ARJ.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/background_images_224/464250.png ) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/background_images_224/empty-road-1.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/background_images_224/images150.png)

### Choosing of foreground images
For foreground images the size is 100 * 100. All the images are of cars. The images are made transparent using GIMP software. Some of the transparent png images of car is downloaded from various sources in internet. The number of these foreground images are 100 of size 100 * 100. The images are as follows-

![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/foreground_images_100/image24.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/foreground_images_100/images10.png ) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/foreground_images_100/maruti-suzuki-car-india-car.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/foreground_images_100/24-249294_ktm-1290-super-duke-r-motorcycle-bike-png.png)

These foreground images are flipped randomly.

### Masked images of foreground images
Each foreground images are masked using GIMP software and it looks as follows-

![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/20-209366_asia-hero-motorcycle-bikes-prices-in-pakistan-super.jpg) 
![alt-text-2](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/2004-mitsubishi-montero-sport-mitsubishi-pajero-sport-car-mitsubishi-motors-mitsubishi-png-thumbnail.jpg) 
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/2014-jaguar-f-type-coupe-2015-jaguar-f-type-r-jaguar-r-coupe-car-jaguar-f-type-white-car-thumbnail.jpg) 
![alt-text-2](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/2015-ferrari-f12berlinetta-sports-car-luxury-vehicle-black-ferrari-f12-berlinetta-car-thumbnail.jpg)

### Overlayed images
Each foreground and flipped foreground images are randomly placed over background images 20 times and it looks as follows-

![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/0.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/100.png ) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/95.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/15.png)

The program which is used to overlay foreground images on background images is [here](https://github.com/futartup/S15A/blob/master/overlay_images.py). The same program is used to generate the mask images which looks as follows:

![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/399244.jpg) 
![alt-text-2](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/399250.jpg) 
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/399278.jpg) 
![alt-text-2](https://github.com/futartup/S15AB/blob/master/raw_images/masked_black_images/399536.jpg)

### Depth images
Each fg_bg images are passed through [this model](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) and created 400K depth images which looks as follows:

![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/depth_bw/0.jpg) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/depth_bw/100.jpg) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/depth_bw/32.jpg) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/depth_bw/20.jpg).

### Total images
- There will be total 400K fg_bg images [here](https://drive.google.com/drive/u/0/folders/1iYDL7k2rHR-2E26U7OCEFDO0b1ybaTiA)
- There will be total 400K mask images [here](https://drive.google.com/drive/u/0/folders/1zHdntBaQzseCn_UX-ppLu32t9sfGYGO9)
- There will be total 400K depth images [here](https://drive.google.com/drive/u/0/folders/1qSgL904seTUoUB3jbaEDY7qqtVRuJTe1)

| Data             | Total Files.        | Mean | Std | Channels |
| ----------------- | ----------- |-------|-----|------|
|          fg_bg   | 400K   | [0.429694425216733, 0.43985525255137686, 0.43281280297561686]| [0.28715867313016924, 0.266891016687173, 0.26731143118502665]| RGB|
|          mask   | 400K   | | | L|
|          depth   | 400K   | | | L|
|          bg   | 100   | | | RGB|

## Model Optimization
### Prunning
[reference](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
Inspired by real world biological neural activity where neurons die itself but creating more synapses. This is a technique which help heavier models to deploy easily making it lightweight. This also gaurentees privacy with private on device computation. However prunning may occur the accuracy to drops but not much. But still if we can be more smart then this drop in accuracy can be made very small. 
But many of the deep learning applications seems to be not using this technique so much according to some of the blogs i read.
Pytorch inbuilt prune class is used in [here](https://github.com/futartup/S15AB/blob/master/train.py). 

Prune ----> Train ----> Test ----> Repeat

```
for e in range(1, self.conf['epochs']):
            print("================================")
            print("Epoch number : {}".format(e))
            if 'prunning' in self.conf['model_optimization'] and self.conf['model_optimization']['prunning']:
                self.conf['model_optimization'].pop('')
                parameters_to_prune = (
                                        (model.down1, 'weight'),
                                        (model.down2, 'weight'),
                                        (model.down3, 'weight'),
                                        (model.down4, 'weight'),
                                        (model.up1, 'weight'),
                                        (model.up2, 'weight'),
                                        (model.up3, 'weight'),
                                        (model.up4, 'weight'),
                                        (model.outc, 'weight'),
                                      )
                prune.global_unstructured(
                                            parameters_to_prune,
                                            pruning_method=prune.L1Unstructured,
                                            amount=0.2,
                                        )
            self.train(e, train_acc, train_loss, train_loss_decrease, global_step_train)
            
            val_loss = self.test(test_acc, tests_loss, test_loss_decrease, global_step_test)
            self.scheduler.step(val_loss)
            print("================================")
```
Print the sparsity of each layer:
```
def sparsity(self):
        print(
                "Sparsity in self.model.down1.weight: {:.2f}%".format(
                    100. * float(torch.sum(self.model.down1.weight == 0))
                    / float(self.model.down1.weight.nelement())
                )
        )
        print(
            "Sparsity in self.model.down2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.down2.weight == 0))
                / float(self.model.down2.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.down3.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.down3.weight == 0))
                / float(self.model.down3.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.down4.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.down4.weight == 0))
                / float(self.model.down4.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up1.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up1.weight == 0))
                / float(self.model.up1.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up2.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up2.weight == 0))
                / float(self.model.up2.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up3.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up3.weight == 0))
                / float(self.model.up3.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.up4.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.up4.weight == 0))
                / float(self.model.up4.weight.nelement())
            )
        )
        print(
            "Sparsity in self.model.outc.weight: {:.2f}%".format(
                100. * float(torch.sum(self.model.outc.weight == 0))
                / float(self.model.outc.weight.nelement())
            )
        )
        print(
            "Global sparsity: {:.2f}%".format(
                100. * float(
                    torch.sum(self.model.down1.weight == 0)
                    + torch.sum(self.model.down2.weight == 0)
                    + torch.sum(self.model.down3.weight == 0)
                    + torch.sum(self.model.down4.weight == 0)
                    + torch.sum(self.model.up1.weight == 0)
                    + torch.sum(self.model.up2.weight == 0)
                    + torch.sum(self.model.up3.weight == 0)
                    + torch.sum(self.model.up4.weight == 0)
                    + torch.sum(self.model.outc.weight == 0)
                )
                / float(
                    self.model.down1.weight.nelement()
                    + self.model.down2.weight.nelement()
                    + self.model.down3.weight.nelement()
                    + self.model.down4.weight.nelement()
                    + self.model.up1.weight.nelement()
                    + self.model.up2.weight.nelement()
                    + self.model.up3.weight.nelement()
                    + self.model.up4.weight.nelement()
                    + self.model.outc.weight.nelement()
                )
            )
        )
```

The model size before prunning is 787.66 MB , which after prunning becomes 430.56 MB with 20% drop of weights.
The weights are dropped using L1 norm, which are lowest 20% connections across the model.

### Dynamic Quantization
This is a technique which involves the conversion of weights and activations of model from float to int, which results in smaller model size according to [pytorch documentation](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html). In the documentation LSTM model is used in experiment. However i have to try this method in my model as well.  



## File Structure
```
├── README.md
├── config-schema.json (Schema file of config to validate the input)
├── config.py (Configuration file)
├── detect_and_remove_duplicate_images.py (Detect and remove duplicate images using hash)
├── find_mean.py (find the mean and standard deviation of images)
├── flip_images.py
├── library  (The main library folder)
│   ├── augmentation
│   │   ├── __init__.py
│   │   └── data_augmenter.py
│   ├── custom_loss.py
│   ├── loader
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── lr_finder.py
│   └── model
│       ├── __init__.py
│       ├── get_model.py
│       ├── res_u_net.py
│       ├── resnet18.py
│       └── u_net.py
├── main.ipynb  (Main file to execute in colab)
├── overlay_images.py
├── predict1.py
├── runs
│   └── MDME
│       └── events.out.tfevents.1589873658.AnupGogoi.local.2269.0
├── saved_models
│   ├── __init__.py
│   └── no_depth_epoch_50_2020-05-15\ 13_41_36.061556_14c14ac5-e61a-455e-8f4b-bbc010d1c6c4.pth
└── train.py
```

## About me
My name is Anup Gogoi and i am an computer vision and AI enthusiast. My dream is to develop products which actually augment the intelligence of mankind in specially medical domain. Human brain can do very complex things, But still it will take some time to figure out the best medicine for Coronavirus pandemic, whereas a combined effort of human and artificial intelligence can do that in very less time.

My github repo is public and collaborators, suggestions are always welcomed.

## Reference

```
@article{futartup,
  author    = {Anup Gogoi},
  title     = {Mask and Monocular Depth Estimation},
  year      = {2020}
}

```



