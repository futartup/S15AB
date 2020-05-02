from imutils import paths
import numpy as np
import argparse
import cv2
import os


def dhash(image, hashSize=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize+1, hashSize))

    diff = resized[:, 1:] > resized[:, :-1]

    return sum([2 ** i for (i,v) in enumerate(diff.flatten()) if v])

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
                help="whether or not duplicates should be removed")
args = vars(ap.parse_args())


print("[INFO] computing images hashes...")
image_paths = list(paths.list_images(args["dataset"]))
sliced_image_paths = image_paths[101:]

# removed rest of the images
for i in sliced_image_paths:
    os.remove(i)

hashes = {}

for x in image_paths:
    image = cv2.imread(x)
    h = dhash(image)

    p = hashes.get(h, [])
    p.append(x)
    hashes[h] = p

count = 0
# Loop over all the hashes of images
for h, hashed_paths in hashes.items():
    print(len(hashed_paths))
    count += 1
    print(count)
    if len(hashed_paths) > 1:
        if args["remove"] <= 0:
            montage = None

            for p in hashed_paths:
                image = cv2.imread(p)
                image = cv2.resize(image, (224, 224))

                if montage is None:
                    montage = image
                else:
                    montage = np.hstack([montage, image])

            print("[INFO] hash: {}".format(h))
            cv2.imshow("Montage", montage)
            cv2.waitKey(3000)
        else:
            for p in hashed_paths[1:]:
                os.remove(p)

for h, hashed_paths in hashes.items():
    print(len(hashed_paths))