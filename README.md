# Object Detector ( S15A and S15 )
[paper implementation U-Net](https://arxiv.org/abs/1505.04597)
This repo is to detect cars in video. The goal is to segment vehicle in images as well as monocular depth estimation. I have created my own dataset.

## Acknowledgements

I would like to thanks all the EVA4 telegram batch members who has helped me to implement the code whenever i am stuck.

I would also like to thank www.theschoolofai.in to give me this opporthunity to get hands on AI. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- GIMP

## Configuration file

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

## To train the model
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
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/output.jpg)


## Constructing the model architecture
To solve this problem i have implemented UNet with total trainable params of 31, 043, 521.
The model code can be found [here](https://github.com/futartup/S15AB/blob/master/library/model/u_net.py)
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/unet.png)

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

## Loss curves
Train loss curve for BCEWithLogitsLoss, ReduceLROnPlateau, Adam

![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/train_loss%20(1).jpg)

Test Loss
![alt-text-1](https://github.com/futartup/S15AB/blob/master/raw_images/test_loss%20(1).jpg)


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
|          fg_bg   | 400K   | | | RGB|
|          mask   | 400K   | | | L|
|          depth   | 400K   | | | L|
|          bg   | 100   | | | RGB|


### Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
|       |   UNet   | 90%.   |

## File Structure
- [library](https://github.com/futartup/S15AB/tree/master/library) (The library code)
  - [augmentation](https://github.com/futartup/S15AB/tree/master/library/augmentation) (The augmentation code)
  - [loader](https://github.com/futartup/S15AB/tree/master/library/loader) (The dataloader code)
  - [model](https://github.com/futartup/S15AB/tree/master/library/model) (The model code)
- [saved_models](https://github.com/futartup/S15AB/tree/master/saved_models) (The models are saved here, though it is large its not stored in here and stored in google drive)
- [data](https://github.com/futartup/S15AB/tree/master/data) (The data for test , train and validation)
- [raw_images](https://github.com/futartup/S15AB/tree/master/raw_images) (All the raw images are stored in here)
- [detect_and_remove_duplicate_images.py](https://github.com/futartup/S15AB/blob/master/detect_and_remove_duplicate_images.py)(Code to delete duplicate images using hash)
- [overlay_images.py](https://github.com/futartup/S15AB/blob/master/overlay_images.py)(Code to produce 800K dataset from fg, bg images)
- [train.py](https://github.com/futartup/S15AB/blob/master/train.py)(The training code and the main file, this is the entry point)
- [predict1.py](https://github.com/futartup/S15AB/blob/master/predict1.py)(Code to predict the output of model)





## About me
My name is Anup Gogoi and i am an computer vision and AI enthusiast. My dream is to develop products which actually augment the intelligence of mankind in specially medical domain. Human brain can do very complex things, But still it will take some time to figure out the best medicine for Coronavirus pandemic, whereas a combined effort of human and artificial intelligence can do that in very less time.

My github repo is public and collaborators, suggestions are always welcomed.






