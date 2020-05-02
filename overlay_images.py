from PIL import Image, ImageOps
from imutils import paths
import argparse
import random
import cv2
import os
import numpy as np
from tqdm import tqdm
import gc


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

count = 0
for x in tqdm(total_flipped_foreground):
    # open the image which we will overlay
    image_foreground = Image.open(x)
    image_foreground_copy = image_foreground.copy()
    all_white_image_foreground_copy = image_foreground_copy.convert()
    for y in image_paths_background:
        image_background_main = Image.open(y)
        for i in range(0, 20):
            # we will write the logic here to overlay images
            image_background_copy = image_background_main.copy()
            x, y = random.randint(0,
                                  max(0, image_background_copy.size[0] - image_foreground.size[0])), \
                                  random.randint(0, max(0,image_background_copy.size[1] - image_foreground.size[1]))
            image_background_copy.paste(image_foreground, (x, y), image_foreground)
            image_background_copy.save(os.getcwd() + '/raw_images/masked_images/{}.png'.format(count))

            # create a black image and overlay it

            img = Image.new('1', (224, 224))
            img.paste(all_white_image_foreground_copy, (x, y), all_white_image_foreground_copy)
            img.save(os.getcwd() + '/raw_images/masked_images_blackwhite/bw_{}.png'.format(count))

            image_background_copy.close()
            img.close()
            count += 1



print(count)

# img = Image.open("data_mask_1354_2030.png")
#
# background = Image.open("background_1354_2030.png")
#
# background.paste(img, (0, 0), img)
# background.save('how_to_superimpose_two_images_01.png',"PNG")