'''
author: Andreas Askim Vatne
version: 1.0
date: 04/01/2023
'''

from skimage import feature
import cv2
import numpy as np
from matplotlib import pyplot as plt
class LBP:

    #constructor that takes three inputs, the image, and the subsequent radius and number of neighbours to be used in the LBP calculation
    def __init__(self, img, rad, points): 
        self.im = cv2.imread(img)
        self.image = self.convert_to_grayscale(cv2.imread(img))
        self.radius = rad
        self.intervalpoints = points

    #converts the input image to grayscale using the OpenCV library
    def convert_to_grayscale(self, img):
        gs_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        plt.imshow(gs_image)
        return gs_image
    
    def calculate_lbp(self): 
        lbp_image = feature.local_binary_pattern(self.image, self.intervalpoints, self.radius, method='default')
        cv2.imshow("LBP image", lbp_image)
        return lbp_image

    
    def histogram(self, lbp):
        n_bins = int(lbp.max() + 1) #number of bins in histogram
        hist, bin_edges = np.histogram(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins)) #ravel to convert lbp 2D-array to 1D-array
        # normalize the histogram, such that it sums to one
        hist = hist.astype("float")
        hist /= hist.sum()
        #"+ 1e-7)"
        return hist, n_bins

    def plot_histogram(self, hist, n_bins):
        fig, ax = plt.subplots(figsize =(10, 7))
        ax.hist(hist, bins=n_bins)

        plt.xlabel("Bins")
        plt.ylabel("Frequency")
        plt.title('LBP histogram')
        # Show plot
        plt.show()

    



#lbp = LBP("C:/Users/andre/OneDrive/Skrivebord/Images/unnamed.jpg", 1, 8)
lbp = LBP("C:/Users/andre/OneDrive/Skrivebord/Informatics/Sapienza/Biometrics/Project/TestImages/testimage1.png", 1, 8)
lbp_image = lbp.calculate_lbp()
histogram, n_bins = lbp.histogram(lbp_image)
lbp.plot_histogram(histogram, n_bins)






