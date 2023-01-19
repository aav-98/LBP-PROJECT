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
    
    def calculate_lbp_images(self, image_paths): 
        lbp_images = []
        for path in image_paths:
            image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_RGB2GRAY)  #converts the input image to grayscale using the OpenCV library
            lbp_image = feature.local_binary_pattern(image, self.intervalpoints, self.radius, method='default')
            lbp_images.append(lbp_image)
            cv2.imshow("LBP-image", lbp_image)
        return lbp_images

    def getHistograms(self, lbp_images):
        lbp_histograms = []
        for lbp_image in lbp_images:
            print(lbp_image)
            hist = self.histogram(lbp_image)
            lbp_histograms.append(hist)
        return lbp_histograms

    def histogram(self, lbp_image):
        hist = []
        tile_histograms = []
        lbp_image_tiles = self.split_lbp_image(lbp_image)
        for lbp_tile in lbp_image_tiles:
            n_bins = int(lbp_image.max() + 1) #number of bins in histogram
            tile_hist, _ = np.histogram(lbp_tile.ravel(), density=True, bins=n_bins, range=(0, n_bins)) #ravel to convert lbp 2D-array to 1D-array
            # normalize the histogram, such that it sums to one
            tile_hist = tile_hist.astype("float")
            tile_hist /= tile_hist.sum()
            tile_histograms.append(tile_hist)
        for tile_hist in tile_histograms:
            hist = np.concatenate((hist, tile_hist))
        return hist

    #split the lbp image into tiles
    def split_lbp_image(self, lbp_image):
        #image size is 168 * 192 (not on current dataset)
        M = lbp_image.shape[0]//2
        N = lbp_image.shape[1]//2
        tiles = [lbp_image[x:x+M,y:y+N] for x in range(0,lbp_image.shape[0],M) for y in range(0,lbp_image.shape[1],N)]
        return tiles

    def plot_histogram(self, lbp_histograms):
        for histogram in lbp_histograms:
            fig, ax = plt.subplots(figsize =(10, 7))
            ax.hist(histogram, bins=(len(histogram)+1))
            plt.xlabel("Bins")
            plt.ylabel("Frequency")
            plt.title('LBP histogram')
            # Show plot
            plt.show()
