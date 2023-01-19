'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       19-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

from project_code import lbp
from project_code import useful_methods as um

#imports machine learning library SVM for classification
from sklearn import svm

#import argparse to handle command-line arguments
import argparse

import numpy as np

def main():

    #parses arguments provided by user in command line
    ap = argparse.ArgumentParser(description="Program runs the LBP algorithm and recognizes classes based on face images from datasets: ") 
    ap.add_argument("-d", "--dataset", dest="dataset", required=True, help="path to the dataset images")
    ap.add_argument("-r", "--radius", dest="radius", required=True, type=int, help="specify the radius to be used for lbp-descriptor")
    ap.add_argument("-n", "--neighbours", dest="neighbours", required=True, type=int, help="specify the number of interval points for lbp-descriptor")
    ap.add_argument("-s", "--save_images", dest="save", default=False, help="select whether or not lbp_image should be saved")
 
    arguments = ap.parse_args()

    # retrieve dataset
    imageseries = um.getDataset(arguments.dataset) #returns dataset in tabular-form (dataframe)

    image_paths = np.asarray(imageseries["image_paths"]) #convert pandas dataframe column image_paths to numpy array
    
    lbp_descriptor = lbp.LBP(arguments.radius, arguments.neighbours) #create an instance of the lbp-descriptor

    lbp_images = lbp_descriptor.calculate_lbp_images(image_paths) #create lbp-images

    imageseries["lbp_images"] = lbp_images #add lbp-images to dataframe

    #save LBP images in dataset-folder if argument is given to -s
    if arguments.save:
        um.saveLBPImages(imageseries, arguments.dataset)

    #calculate lbp-histograms
    lbp_historgrams = lbp_descriptor.getHistograms(lbp_images)
    imageseries["lbp_histograms"] = lbp_historgrams
    print(imageseries["lbp_histograms"])

    #view lbp-histograms
    lbp_descriptor.plot_histogram(lbp_historgrams)

    # train a linear SVM on the training data

    """ model = svm.LinearSVC(C=100.0, random_state=42)
    model.fit(lbp_histograms, classes) #fit the model according to training data (lbp-histograms and connected class-labels) """

main()