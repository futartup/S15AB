from PIL import Image, ImageOps, ImageStat
from imutils import paths
import argparse
import random
import cv2
import os
import numpy as np
from tqdm import tqdm
import gc
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile


CHANNEL_NUM = 3
pixel_num = 0 # store all pixel number in the dataset
channel_sum = np.zeros(CHANNEL_NUM)
channel_sum_squared = np.zeros(CHANNEL_NUM)

ap = argparse.ArgumentParser()
ap.add_argument("-d1", "--dataset1", required=True,
                help="path to background dataset")
ap.add_argument("-d2", "--dataset2", required=True,
                help="path to flipped foreground dataset")
ap.add_argument("-d3", "--dataset3", required=True,
                help="path to foreground dataset")
ap.add_argument("-o", "--output_dir", required=False,
                help="path to output directory")
args = vars(ap.parse_args())


print("[INFO] processing the images...")
image_paths_background = list(paths.list_images(args["dataset1"]))
image_paths_flipped_foreground = list(paths.list_images(args["dataset2"]))
image_paths_foreground = list(paths.list_images(args["dataset2"]))
total_flipped_foreground = image_paths_flipped_foreground + image_paths_foreground


for x, image in enumerate(tqdm(total_flipped_foreground)):
    # open the image which we will overlay
    image_foreground = Image.open(x)
    image_foreground_copy = image_foreground.copy()
    all_white_image_foreground_copy = image_foreground_copy.convert('1')
    for y in image_paths_background:
        image_background_main = Image.open(y)
        for i in range(0, 20):
            # we will write the logic here to overlay images
            image_background_copy = image_background_main.copy()
            x, y = random.randint(0,
                                  max(0, image_background_copy.size[0] - image_foreground.size[0])), \
                                  random.randint(0, max(0,image_background_copy.size[1] - image_foreground.size[1]))
            image_background_copy.paste(image_foreground, (x, y), image_foreground)
            image_background_copy = image_background_copy.convert('RGB')
            image_background_copy.save(os.getcwd() + '/raw_images/masked_images/{}.jpg'.format(count))

            # create a black image and overlay it

            img = Image.new('1', (224, 224))
            img.paste(all_white_image_foreground_copy, (x, y), all_white_image_foreground_copy)
            img.save(os.getcwd() + '/raw_images/masked_images_blackwhite/bw_{}.png'.format(count))

            # Calculate the mean and std for each image
            im = np.asarray(image_background_copy)
            im = im / 255.0
            pixel_num += (im.size / CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

            image_background_copy.close()
            img.close()
            count += 1


bgr_mean = channel_sum / pixel_num
bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

# change the format from bgr to rgb
rgb_mean = list(bgr_mean)[::-1]
rgb_std = list(bgr_std)[::-1]

print(rgb_mean, rgb_std)
