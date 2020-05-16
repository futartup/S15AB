# EVA Assignment Submissions ( S15A and S15 )

This is an original submission of assignments from S15A and S15. The goal is to segment and create a bounding box around the object detected as well as monocular depth estimation. I have created my own dataset.

## Acknowledgements

I would like to thanks all the EVA4 telegram batch members who has helped me to implement the code whenever i am stuck.

I would also like to thank www.theschoolofai.in to give me this opporthunity to get hands on AI. 

## Prerequisites
- Python 3.6+
- PyTorch 1.0+
- GIMP

## Constructing the model architecture
To solve this problem i have implemented UNet with total trainable params of 31, 043, 521.
The model code can be found [here](https://github.com/futartup/S15AB/blob/master/library/model/u_net.py)

## Representation of task
The goal is to segment the cars(which is the case in my case) in an image and predict the depth in an image. Simply, my goal is to take either a RGB color image (height×width×3) or a grayscale image (height×width×1) and output a segmentation map where each pixel contains a class label represented as an integer (height×width×1).
![alt-text-1](https://github.com/futartup/S15AB/blob/master/data/images/lady.png) 

## Loss functions
I have used CrossEntropyLoss for mask prediction and MSELoss for depth estimation.
These loss functions can be defines in [here](https://github.com/futartup/S15AB/blob/master/config.py)

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
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/masked_images_100/2013-toyota-corolla-le-car-toyota-sienna-front-wheel-drive-toyota-thumbnail.jpg ) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/masked_images_100/2016-land-rover-range-rover-sport-2017-land-rover-range-rover-sport-car-rover-company-rang-thumbnail.jpg) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/masked_images_100/bmw-png-transparent-images-169843-174756.jpg)

### Overlayed images
Each foreground and flipped foreground images are randomly placed over background images 20 times and it looks as follows-

![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/0.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/100.png ) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/95.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/fg_bg_images/15.png)

The program which is used to overlay foreground images on background images is [here](https://github.com/futartup/S15A/blob/master/overlay_images.py). The same program is used to generate the mask images which looks as follows:

![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/mask/bw_0.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/mask/bw_2.png) 
![alt-text-1](https://github.com/futartup/S15A/blob/master/raw_images/mask/bw_95.png) 
![alt-text-2](https://github.com/futartup/S15A/blob/master/raw_images/mask/bw_99.png).

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


### Results and some observations S15A submission


### Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
|       |      |

## File Structure


## About me
My name is Anup Gogoi and i am an computer vision and AI enthusiast. My dream is to develop products which actually augment the intelligence of mankind in specially medical domain. Human brain can do very complex things, But still it will take some time to figure out the best medicine for Coronavirus pandemic, whereas a combined effort of human and artificial intelligence can do that in very less time.

My github repo is public and collaborators, suggestions are always welcomed.






