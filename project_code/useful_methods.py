'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       19-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

#import os to be able to traverse directories and files in operating system
import os

#import pandas to organize and extract information from datasets
import pandas as pd

from PIL import Image
import numpy as np

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

def saveLBPImages(imageSeries:pd.DataFrame, path_segment):
    for index in imageSeries.index:
        lbp_image = imageSeries["lbp_images"][index]
        print(os.getcwd())
        path = os.getcwd() + "/" + path_segment + "/" + imageSeries["image_labels"][index] + ".jpg"
        img = Image.fromarray(lbp_image.astype("uint8"), 'L')
        img.save(path)