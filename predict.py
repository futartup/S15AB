import argparse
from library.model.u_net import UNet
from PIL import Image
from imutils import paths
import logging
import torch
from library.loader.data_loader import DepthDataSet
from torchvision import transforms


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


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


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
    ap.add_argument("--data_dir", required=True,
                    help="The Directory to the data")
    ap.add_argument("--load_model", required=False,
                    help="Load the saved model")
    args = vars(ap.parse_args())
    
    net = UNet(3,2)
    logging.info("Loading model {}".format(args['model']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args['model'], map_location=device))
    logging.info("Model loaded !")

    images = list(paths.list_images(args["input_dir"]))

    for i, image in enumerate(images):
        filename = image.split('/')[-1]
        logging.info("\nPredicting image {} ...".format(img))

        img = Image.open(image)

        mask = predict_img(net=net,
                           full_img=img,
                           device=device)

        result = mask_to_image(mask)
        result.save(args['output'] + '/' + filename)

