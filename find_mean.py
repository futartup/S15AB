from PIL import Image
from imutils import paths
import argparse
import os
import numpy as np
from tqdm import tqdm


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to the data dir where fg_bg, bg, mask and depth images are there")
args = vars(ap.parse_args())

fg_bg_test = args['dataset'] + '/fg_bg/test'
fg_bg_train = args['dataset'] + '/fg_bg/train'

# bg_test = args['dataset'] + '/bg/test'
# bg_train = args['dataset'] + '/bg/train'

mask_test = args['dataset'] + '/mask/test'
mask_train = args['dataset'] + '/mask/train'

depth_test = args['dataset'] + '/depth/test'
depth_train = args['dataset'] + '/depth/train'

result = {}

print("[INFO] finding the mean of images...")

for x in [fg_bg_test, fg_bg_train, mask_test, mask_train, depth_test, depth_train]:
    images = list(paths.list_images(x))
    first_image = Image.open(images[0])
    if first_image.mode == 'RGB':
        CHANNEL_NUM = 3
    else:
        CHANNEL_NUM = 1
    first_image.close()
    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    for pos, image in enumerate(tqdm(images)):
        img = Image.open(image)

        # Calculate the mean and std for each image
        im = np.asarray(img)
        im = im / 255.0
        pixel_num += (im.size / CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

        img.close()
        
    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    mean = list(bgr_mean)[::-1]
    std = list(bgr_std)[::-1]

    result[x] = [mean, std]


print(result)

