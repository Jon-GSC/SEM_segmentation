# general processing sem image by Jon on 2018-08-21, train-predict
# cnn works after adjust learning_rate etc parameters, estimator.train used and work good.
#
#
import os
import cv2
import numpy as np
import pandas as pd
# import collections
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import math
# from scipy import ndimage
# from itools import image_tools as it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'    # '0'display  '3'-no display gpu device running information on 20180823
#-----------------------------------------------------------------------------------------------------------------------
# n_pix = 128

#-----------------------------------------------------------------------------------------------------------------------
def img_cut(index,n_pix,current_path,datasets_path,n_hw):   # cut image for training dataset folder on 2018-08-15.
    # xy_radium = xy_radium
    os.makedirs(datasets_path, exist_ok=True)
    xy_radium = int(n_hw[0] / 2)
    for file in os.listdir(current_path):
        if file.endswith(".tif"):

            image = cv2.imread(os.path.join(current_path, file), cv2.IMREAD_GRAYSCALE)
            if index ==1:
                image[image != 255] = 0
            img_size = image.shape

            icount = 0
            y_centre = int(n_hw[0]/2)
            while y_centre < img_size[0]-xy_radium+1:
                y1 = y_centre - xy_radium
                y2 = y_centre + xy_radium
                x_centre = xy_radium
                while x_centre < img_size[1]-xy_radium+1:
                    x1 = x_centre - xy_radium
                    x2 = x_centre + xy_radium
                    img = image[x1:x2, y1:y2]
                    # print(img.shape)
                    # exit()
                    filename, ext = os.path.splitext(file)
                    name = "{}/{}_{:04}.tif".format(datasets_path,filename, icount)
                    cv2.imwrite(name, img)   #save
                    icount += 1
                    x_centre += n_pix
                y_centre += n_pix


