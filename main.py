'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       12-02-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

#import necessary classes and methods from files in project folder
from project_code import lbp
from project_code import useful_methods as um

#import machine learning libraries for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#import joblib to save and load classifier model from memory
import joblib

#import argparse to handle command-line arguments
import argparse

import numpy as np

#import scipy spatial distance matrix to compute distance matrix
from scipy.spatial import distance_matrix

#import pyplot to plot values from evaluation
import matplotlib.pyplot as plt

import time
import os

def main():

    #parses arguments provided by user in command line
    ap = argparse.ArgumentParser(description="Program runs the LBP algorithm and recognizes classes based on face images from datasets: ") 
    ap.add_argument("-d", "--dataset", dest="dataset", required=True, help="path to the dataset images")
    ap.add_argument("-r", "--radius", dest="radius", required=True, type=float, help="specify the radius to be used for lbp-descriptor")
    ap.add_argument("-p", "--sampling_points", dest="samplingpoints", required=True, type=int, help="specify the number of interval points for lbp-descriptor")
    ap.add_argument("-s", "--save_images", dest="save", default=False, help="select whether or not lbp_image should be saved")
    ap.add_argument("-hq", "--hist_equal", dest="histeq", default=False, help="select whether or not to apply histogram equalization to images")
    ap.add_argument("-m", "--method", dest="method", default=False, help="method to determine the pattern, select between default, ror, uniform, nri_uniform, and var")
    ap.add_argument("-t", "--training", dest="training", default=False, help="select whether or not to train ML model")
    ap.add_argument("-he", "--height", dest="height", default=4, type=int, help="select the number of image splits on the vertical side of the image")
    ap.add_argument("-w", "--width", dest="width", default=4, type=int, help="select the number of image splits on the horizontal side of the image")
 
    arguments = ap.parse_args()

    # retrieve dataset
    imageseries = um.getDataset(arguments.dataset) #returns dataset in tabular-form (dataframe)

    image_paths = np.asarray(imageseries["image_paths"]) #convert pandas dataframe column image_paths to numpy array
    
    lbp_descriptor = lbp.LBP(arguments.radius, arguments.samplingpoints) #create an instance of the lbp-descriptor

    start_time = time.time() #record start time from when program has retrieved dataset until it has printed cumulative match curve scores

    lbp_images = lbp_descriptor.calculate_lbp_images(image_paths, arguments.histeq, arguments.method) #create lbp-images

    imageseries["lbp_images"] = lbp_images #add lbp-images to dataframe

    #save LBP images in dataset-folder if argument is given to -s
    #if (arguments.save and arguments.method == "default"):
    if (arguments.save):
        um.saveLBPImages(imageseries, arguments.dataset)

    print("Start calculating lbp-histograms", "\n")
    #calculate lbp-histograms
    lbp_histograms = lbp_descriptor.getHistograms(lbp_images, arguments.height, arguments.width)

    X = lbp_histograms #feature-vectors
    y = np.asarray(imageseries["class_labels"]) #class labels

    #sample 60 percent of the data for training and hold out 40 percent for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    if arguments.training:  #if training is passed as argument

        print("Training of nearest neighbor classification model has begun", "\n")

        #create an instance of the k-nearest neighbour classfier
        clf = KNeighborsClassifier(n_neighbors= 3)

        clf.fit(X_train, y_train) #fit the model according to training data (lbp-histograms and related class-labels)
        
        joblib.dump(clf, "clf_model.pkl")

    clf = joblib.load("clf_model.pkl")  #load classifier from memory

    print("Classification of test images has started:", "\n")
    y_pred = clf.predict(X_test)
    print("Prediction:", y_pred, "\n")
    print("Correct class labels: ", y_test, "\n")

    #accuracy score provides a float between 0 and 1  representing the number of test samples that were correctly classified by the k-nearest neighbour classifier
    print("Accuracy score:", str(accuracy_score(y_test, y_pred)), "\n")

    #returns a distance matrix of all pairwise distances between histograms in training samples and test samples
    dist_matrix = distance_matrix(X_test, X_train)

    #calculate the recogntion rate
    recog_rate_at_rank_1 = um.calculate_recognition_rate_at_rank_1(dist_matrix, y_train, y_test)

    #calculate the correct matches at each rank, stopping for every row when a correct match is identified
    matches_at_every_rank = um.match_scores(dist_matrix, y_train, y_test)
    print(matches_at_every_rank, "\n")

    #calculate the cumulative match curve scores to plot cmc
    cumulative_match_curve_scores = um.cumulative_match_curve_scores(matches_at_every_rank, len(y_test))
    print(cumulative_match_curve_scores, "\n")

    print("--- %s seconds ---" % (time.time() - start_time), "\n")

    #using matplotlib to visually representent the recognition rate and the cumulative match curve
    plt.plot([1], [recog_rate_at_rank_1], 'ro')
    plt.plot(range(1, dist_matrix.shape[1] + 1), cumulative_match_curve_scores, 'g-')
    plt.xlabel('Rank')
    plt.ylabel('Probability of Identification')
    plt.title('Recognition Rate at Rank 1 and Cumulative Match Curve')
    #save the cmc graph in a separate folder
    filename = (os.getcwd() + "/cmc_graphs/" + "LBP_" + str(int(arguments.radius)) + "_" 
    + str(arguments.samplingpoints) + "_" + str(arguments.height) + "_" + str(arguments.width) +".png")
    plt.savefig(filename, format="png", dpi=300)
    plt.show()

    print("Recognition rate: ", recog_rate_at_rank_1, "\n")

main()