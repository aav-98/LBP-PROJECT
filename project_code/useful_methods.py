'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       19-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

#import os to be able to traverse directories and files in operating system
import os
from matplotlib import pyplot as plt

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
        path = os.getcwd() + "/" + path_segment + "/" + imageSeries["image_labels"][index] + ".jpg"
        img = Image.fromarray(lbp_image.astype("uint8"), 'L')
        img.save(path)

#split the lbp image into tiles
def split_lbp_image(lbp_image):
    #image size is 168 * 192 (not on current dataset)
    M = lbp_image.shape[0]//4  #shape[0] gives image height
    N = lbp_image.shape[1]//4   #shape[1] gives image width
    tiles = [lbp_image[x:x+M,y:y+N] for x in range(0,lbp_image.shape[0],M) for y in range(0,lbp_image.shape[1],N)]
    return tiles

def plot_histogram(lbp_histograms):
        for histogram in lbp_histograms:
            fig, ax = plt.subplots(figsize =(10, 7))
            ax.hist(histogram, bins=(len(histogram)+1))
            plt.xlabel("Bins")
            plt.ylabel("Frequency")
            plt.title('LBP histogram')
            # Show plot
            plt.show()