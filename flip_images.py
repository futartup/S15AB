from PIL import Image, ImageOps
import argparse
import random
import os, sys


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")

args = vars(ap.parse_args())
   
for filename in os.listdir(args['dataset']):
    for file in os.listdir(os.getcwd() + '/data/Session 2 Dataset/' + filename):    
        file = "{}/data/Session 2 Dataset/{}/{}".format(os.getcwd(), filename, file)
        try:
            img = Image.open(file) # open the image file
            print(img.mode)
            if img.mode in ['RGBA', 'L', 'P']:
                #img = img.convert('RGB')
                print('PNG file:', img.mode)
                #path = img.filename.split('.')[0]
                path_ext = img.filename.split('.')
            
                #filename = img.filename.split('/')[-1].split('.')[0]
                img.load()  # needed for split()
                background = Image.new('RGB', img.size, color=0)
                background.paste(img)  # 3 is the alpha channel
                background.save(path_ext[0]+'.jpg')
                print(background.mode)
                os.remove(file)

            #img.verify() # verify that it is, in fact an image
        except (IOError, SyntaxError, AttributeError) as e:
            print('Bad file:', e) # print out the names of corrupt files
            #os.remove(file)