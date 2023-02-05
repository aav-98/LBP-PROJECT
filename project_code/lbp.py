'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       19-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

from matplotlib import pyplot as plt
from project_code import useful_methods as um

from skimage import feature     #skimage: image processing for python
import cv2     #open-source computer vision library
import numpy as np

class LBP:

    #constructor that takes two inputs, the radius and the number of neighbours to be used in the LBP calculation
    def __init__(self, rad, points): 
        self.radius = rad
        self.intervalpoints = points
    
    def calculate_lbp_images(self, image_paths, histeq, method): 
        lbp_images = []
        for path in image_paths:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)  #converts the input image to grayscale using the OpenCV library
            if histeq:
                image = cv2.equalizeHist(image)
            lbp_image = feature.local_binary_pattern(image, self.intervalpoints, self.radius, method=method)
            lbp_images.append(lbp_image)
        return lbp_images

    def getHistograms(self, lbp_images):
        lbp_histograms = []
        for lbp_image in lbp_images:
            hist = self.histogram(lbp_image)
            lbp_histograms.append(hist)
        return lbp_histograms

    def histogram(self, lbp_image):
        hist = []
        tile_histograms = []
        lbp_image_tiles = um.split_lbp_image(lbp_image)
        for lbp_tile in lbp_image_tiles:
            n_bins = int(lbp_image.max() + 1) #number of bins in histogram
            tile_hist, _ = np.histogram(lbp_tile.ravel(), density=True, bins=n_bins, range=(0, n_bins)) #ravel to convert lbp 2D-array to 1D-array
            # normalize the histogram, such that it sums to one
            #tile_hist = tile_hist.astype("float")
            #tile_hist /= tile_hist.sum()
            tile_histograms.append(tile_hist)
        hist = np.concatenate(tile_histograms)
        return hist