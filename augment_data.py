import numpy as np
import os, json, cv2, random, wget

dataset_path = "/home/saumyas/Projects/IAM-Vision/Vision_based_pose_estimation/Vision_based_pose_estimation/datasets/mug/" 

for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        imfile = dataset_path + filename
        im = cv2.imread(imfile)
        # import ipdb; ipdb.set_trace()
        flipped = cv2.flip(im, 1)
        # cv2.imshow("Normal", im)
        # cv2.imshow("Flipped", flipped)
        cv2.imwrite(dataset_path+ filename[:-4] + "_flip.png", flipped)
