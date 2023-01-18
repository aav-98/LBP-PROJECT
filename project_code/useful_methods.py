'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       18-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

import os
import cv2

import pandas as pd
from matplotlib import pyplot as plt

def getDataset(path_segment):   #returns the dataset partitioned into relevant arrays

    class_labels = []
    image_paths = []
    image_labels = []

    dataset_path = os.getcwd() + "/" + path_segment + "/"   #creates a path to the required directory containing requested dataset images

    for class_directory in os.listdir(dataset_path):
        class_images = os.listdir(os.path.join(dataset_path, class_directory))
        for image in class_images:
            image_paths.append(os.path.join(dataset_path, class_directory, image))
            image_labels.append(image)
            class_labels.append(class_directory)

    data = {"image_paths":image_paths, "image_labels":image_labels, "class_labels":class_labels}
    imageseries = pd.DataFrame(data)    #create a data frame from the data extracted above

    return imageseries

def convert_to_grayscale(img):    #converts the input image to grayscale using the OpenCV library
    gs_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    plt.imshow(gs_image)
    return gs_image
