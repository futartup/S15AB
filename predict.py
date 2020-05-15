import argparse
import cv2
from library.model.u_net import UNet
from PIL import Image
from imutils import paths
import logging
import torch
from library.loader.data_loader import DepthDataSet
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import resize


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(DepthDataSet.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    

    plasma = plt.get_cmap('plasma')

    #shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    
    all_images = []
    count = 0
    for pos, i in enumerate(outputs):
        imgs = []
        
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[pos][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            im = Image.fromarray(np.uint8(rescaled*255))
            im.save('/content/drive/My Drive/Colab Notebooks/S15A-B/Data/final_data/output/{}.jpg'.format(count))
            #imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        #img_set = np.hstack(imgs)
        #all_images.append(img_set)

    #all_images = np.stack(all_images)
    
    #return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def plot_img_and_mask(img, mask, filename):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    #plt.savefig('/content/drive/My Drive/Colab Notebooks/S15AB/data/output/{}'.format(filename))
    plt.savefig('/Users/anupgogoi/Desktop/{}'.format(filename))
def mask_to_image(mask):
    img = Image.fromarray(np.uint8(mask[1]*255))
    #img = Image.fromarray((mask * 255))
    print(img)
    return img

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Predict masks from input images',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", "-m", required=True,
                    default='MODEL.pth',
                    metavar='FILE',
                    help="Specify the file where model is stored")
    ap.add_argument("--input_dir", "-i", required=False,
                    metavar='INPUT', nargs='+',
                    help="The directory of input images")
    ap.add_argument("--output", "-o", 
                    metavar='INPUT', nargs='+', required=False,
                    help="The Directory of output images")
    ap.add_argument("--viz", "-v", required=False, action="store_true",
                    default=False,
                    help="The width of an image")
    ap.add_argument("--data_dir", required=False,
                    help="The Directory to the data")
    ap.add_argument("--load_model", required=False,
                    help="Load the saved model")
    args = vars(ap.parse_args())
    
    net = UNet(3,2, True)
    logging.info("Loading model {}".format(args['model']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    state_dict = torch.load(args['model'], map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v

    net.load_state_dict(new_state_dict, strict=False)
    #net.load_state_dict(torch.load(args['model'], map_location=device))
    logging.info("Model loaded !")
    print(args)
    images = list(paths.list_images(args["input_dir"][0]))
    outputs = []
    #images = ['/content/drive/My Drive/Colab Notebooks/S15A-B/Data/final_data/val/399708.jpg']
    for i, image in enumerate(images):
        filename = image.split('/')[-1]
        logging.info("\nPredicting image {} ...".format(filename))

        img = Image.open(image)

        mask = predict_img(net=net,
                           full_img=img,
                           device=device)
        mask = np.where(mask == True, 1, 0)
        #print(mask)
        #print(type(mask))
        result = mask_to_image(mask)
        
        alpha=0.9
        # img = cv2.imread('/Users/anupgogoi/Desktop/count.jpg')
        # mask = cv2.imread('/Users/anupgogoi/Desktop/392885.jpg')

        mask = cv2.resize(mask, img.shape[1::-1])
        dst = cv2.addWeighted(img, alpha, mask, alpha, 0)

        cv2.imwrite(args['output'] + '/' + filename, dst)
        #plot_img_and_mask(img, mask, filename)
        #result.save(args['output'][0] + filename)

#display_images(outputs, is_colormap=True)