'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       18-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

#imports lbp descriptor code from the 'code' module
from project_code import lbp
from project_code import useful_methods as um

'''
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection
Source: https://scikit-learn.org/stable/modules/svm.html#svm-classification
'''
#imports machine learning library SVM for classification
from sklearn import svm

#import other useful libraries
import argparse
import cv2
import os

#import pandas to organize and extract information from datasets 
import pandas as pd

import numpy as np

def main():

    #parses arguments provided by user in command line
    ap = argparse.ArgumentParser(description="Program runs the LBP algorithm and recognizes classes based on face images from database: ") 
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset images")
    ap.add_argument("-r", "--radius", required=True, help="specify the radius to be used for lbp-descriptor")
    ap.add_argument("-n", "--neighbours", required=True, help="specify the number of interval points for lbp-descriptor")

    arguments = vars(ap.parse_args())

    # retrieve dataset
    imageseries = um.getDataset(arguments["dataset"])

    #create lbp images
    image_paths = np.asarray(imageseries["image_paths"])    #convert pandas data frame column to numpy array

    try:
        radius = int(arguments["radius"])
        neighbours = int(arguments["neighbours"])
        lbp_descriptor = lbp.LBP(radius, neighbours)  #create an instance of the lbp-descriptor
    except:
        #create an instance of the lbp-descriptor with standard inputs if error converting cmd-line input to intgers
        lbp_descriptor = lbp.LBP(1, 8)

    lbp_images = lbp_descriptor.calculate_lbp_images(image_paths)
    imageseries["lbp_images"] = lbp_images

    #calculate lbp-histograms
    lbp_historgrams = lbp_descriptor.histogram(lbp_images)
    imageseries["lbp_histograms"] = lbp_historgrams
    print(imageseries)

    # train a linear SVM on the training data

    """ model = svm.LinearSVC(C=100.0, random_state=42)
    model.fit(lbp_histograms, classes) #fit the model according to training data (lbp-histograms and connected class-labels) """

main()