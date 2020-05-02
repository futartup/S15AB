from PIL import Image, ImageOps
from imutils import paths
import argparse
import random
import cv2
import os, sys

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-rs", "--image_resize", required=False,
                help="The size to which the image to be resized")
ap.add_argument("-o", "--output_dir", required=False,
                help="The output directory to save the files")
ap.add_argument("-f", "--flip", required=False,
                help="Flip the images")
args = vars(ap.parse_args())

print("[INFO] processing the images...")
image_paths = list(paths.list_images(args["dataset"]))

for path_image in image_paths:
    image = Image.open(path_image)
    file_name = path_image.split('/')[-1]
    if "image_resize" in args and args["image_resize"] is not None:
        resize_size = int(args["image_resize"])
        image = image.resize((resize_size, resize_size))
    if "flip" in args and args["flip"] is not None:
        rotate = random.randint(0, 360)
        image = image.rotate(rotate)
    if "output_dir" in args and args["output_dir"] is not None:
        image.save(args["output_dir"] + '/' + file_name)
    else:
        image.save(os.getcwd() + '/foreground_images/' + file_name)