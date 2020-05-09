from PIL import Image, ImageOps, ImageStat
from imutils import paths
import argparse
import random
import cv2
import os
import numpy as np
from tqdm import tqdm
import gc

def main(d1, d2, o):
    fg_bg_images = list(paths.list_images(d1))

    for x, image_path in enumerate(tqdm(fg_bg_images)):
        file_name = image_path.split('/')[-1].split('.')[0]

        fg_bg = Image.open(image_path).convert('1')
        print(fg_bg.size, fg_bg.mode)
        fg_bg_mask = Image.open(d2 + '/bw_{}.png'.format(file_name))
        print(fg_bg_mask.size, fg_bg_mask.mode)
        
        fg_bg_np = np.array(fg_bg)
        fg_bg_mask_np = np.array(fg_bg_mask)

        #pil_image = Image.merge('L', (fg_bg, fg_bg_mask))
        merge_image = cv2.vconcat([fg_bg_np, fg_bg_mask_np])
        print(merge_image.shape)
        #pil_image = Image.fromarray(merge_image)
        cv2.imwrite('./S15A-B/data/train/' + 'final_{}.jpg'.format(file_name), merge_image)
        break

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d1", "--dataset1", required=True,
                    help="path to background dataset")
    ap.add_argument("-d2", "--dataset2", required=True,
                    help="path to flipped foreground dataset")
    ap.add_argument("-o", "--output_dir", required=False,
                    help="path to output directory")
    args = vars(ap.parse_args())
    main(args['dataset1'], args['dataset2'], args['output_dir'])
