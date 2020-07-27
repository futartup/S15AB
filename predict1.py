import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
from library.model.mobilenet_v2 import mobilenet_v2
from library.loader.data_loader import DepthDataSet
from library.augmentation.data_augmenter import TransfomedDataSet


def plot_img_and_mask(img, mask, filename):
    
    fig, ax = plt.subplots(1, 3)
    indices_list = np.where(np.all(mask != False, axis=-1))
    print("excluded indices are", indices_list)
    mask[indices_list] = 0
    print(mask)
    cv2.imwrite('/content/drive/My Drive/Colab Notebooks/S15AB/m.jpg', np.uint8(cm.gist_earth(mask))*255) 
    # mask_black = np.any(mask != 1, axis=-1)
    # mask_white = np.any(mask != 0, axis=-1)
    # #mask1 = np.where(mask[0 , :, :]==True, 1, 0)
    # #mask2 = np.where(mask[1 , :, :]==True, 1, 0)
    # print(mask_white.shape)
    # print(mask_black.shape)
    # im = Image.fromarray(np.uint8(cm.gist_earth(mask_white))*255).convert('RGB')
    # #im2 = Image.fromarray(np.uint8(cm.gist_earth(mask2))*255).convert('RGB')
    # im.save('/content/drive/My Drive/Colab Notebooks/S15AB/{}'.format(filename))
    #im2.save('/content/drive/My Drive/Colab Notebooks/S15AB/{}'.format("mask2.jpg"))
    # ax[0].imshow(img)
    # ax[1].imshow(mask[0 , :, :])
    #ax[2].imshow(mask[1 , :, :])
    #plt.show()
    #classes = mask.shape[2] if len(mask.shape) > 2 else 1
    #fig, ax = plt.subplots(1, 3)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.imshow(img)

    # plt.subplot(212)
    # plt.imshow(mask[0 , :, :])

    # plt.subplot(222)
    # plt.imshow(mask[1 , :, :])

    # ax[0].set_title('Input image')
    # ax[0].imshow(img)
    # ax[1].set_title(f'Output firt mask')
    # ax[1].imshow(np.where(mask[0 , :, :]==True, 1/255, 0))
    # ax[2].set_title(f'Output second mask')
    # ax[2].imshow(np.where(mask[1 , :, :]==True, 1/255, 0))
    # plt.xticks([]), plt.yticks([])
    #plt.savefig('/content/drive/My Drive/Colab Notebooks/S15AB/{}'.format(filename))

def predict_img(net,
                full_img,
                device,
               ):
    net.eval()
    transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
    ])
    #image = Image.open(io.BytesIO(image_bytes))
    #print('image opened')
    img = transformations(full_img).unsqueeze(0)
    #img = torch.from_numpy(DepthDataSet.preprocess(full_img, scale_factor))

    # img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        result = output.argmax().item()
    
    return result


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='The full path to the input image', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    net = mobilenet_v2()

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    state_dict = torch.load(args.model, map_location=device)
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = 'module.'+k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict[k]=v
        
    net.load_state_dict(state_dict, strict=False)

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        result = predict_img(net=net,
                           full_img=img,
                           device=device)
        print(result)
        # if not args.no_save:
        #     out_fn = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_files[i])

        #     logging.info("Mask saved to {}".format(out_files[i]))

        # if args.viz:
        #     logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        #     plot_img_and_mask(img, mask, 'mask.jpg')
