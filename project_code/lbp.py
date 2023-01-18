'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       18-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

from project_code import useful_methods as um

from skimage import feature     #skimage: image processing for python
import cv2     #open-source computer vision library
import numpy as np
from matplotlib import pyplot as plt

class LBP:

    #constructor that takes two inputs, the radius and the number of neighbours to be used in the LBP calculation
    def __init__(self, rad, points): 
        self.radius = rad
        self.intervalpoints = points
    
    def calculate_lbp_images(self, image_paths): 
        lbp_images = []
        for path in image_paths:
            print(path)
            image = um.convert_to_grayscale(cv2.imread(path))
            lbp_image = feature.local_binary_pattern(image, self.intervalpoints, self.radius, method='default')
            lbp_images.append(lbp_image)
            cv2.imshow("LBP image", lbp_image)
        return lbp_images

    
    def histogram(self, lbp_images):
        lbp_histograms = []
        for lbp_image in lbp_images:
            n_bins = int(lbp_image.max() + 1) #number of bins in histogram
            hist, bin_edges = np.histogram(lbp_image.ravel(), density=True, bins=n_bins, range=(0, n_bins)) #ravel to convert lbp 2D-array to 1D-array
            # normalize the histogram, such that it sums to one
            hist = hist.astype("float")
            hist /= hist.sum()
            lbp_histograms.append(hist)
        return lbp_histograms

"""     def plot_histogram(self, hist, n_bins):
        fig, ax = plt.subplots(figsize =(10, 7))
        ax.hist(hist, bins=n_bins)

        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.title('LBP histogram')
        # Show plot
        plt.show() 
 
histogram, n_bins = lbp.histogram(lbp_image)
lbp.plot_histogram(histogram, n_bins) """






