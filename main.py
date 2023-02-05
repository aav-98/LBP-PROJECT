'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       19-01-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

from project_code import lbp
from project_code import useful_methods as um

#imports machine learning libraries for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


#import argparse to handle command-line arguments
import argparse

import numpy as np

from scipy.spatial import distance_matrix

#import pandas to organize and extract information from datasets
import pandas as pd

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
 
    arguments = ap.parse_args()

    # retrieve dataset
    imageseries = um.getDataset(arguments.dataset) #returns dataset in tabular-form (dataframe)

    image_paths = np.asarray(imageseries["image_paths"]) #convert pandas dataframe column image_paths to numpy array
    
    lbp_descriptor = lbp.LBP(arguments.radius, arguments.samplingpoints) #create an instance of the lbp-descriptor

    lbp_images = lbp_descriptor.calculate_lbp_images(image_paths, arguments.histeq, arguments.method) #create lbp-images

    imageseries["lbp_images"] = lbp_images #add lbp-images to dataframe

    #save LBP images in dataset-folder if argument is given to -s
    if (arguments.save and arguments.method == "default"):
        um.saveLBPImages(imageseries, arguments.dataset)

    print("Start calculating lbp-histograms: ")
    #calculate lbp-histograms
    lbp_histograms = lbp_descriptor.getHistograms(lbp_images)

    #view lbp-histograms
    #um.plot_histogram(lbp_historgrams)

    X = lbp_histograms #feature-vectors
    y = np.asarray(imageseries["class_labels"]) #class labels

    #sample 60 percent of the data for training and hold out 40 percent for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

    if arguments.training:  #if training is passed as argument

        print("Training of nearest neighbor classification model has begun")

        #create an instance of the k-nearest neighbour classfier
        clf = KNeighborsClassifier(n_neighbors= 3)

        clf.fit(X_train, y_train) #fit the model according to training data (lbp-histograms and connected class-labels)
        
        joblib.dump(clf, "clf_model.pkl")

    clf = joblib.load("clf_model.pkl")  #load classifier from memory

    print("Classification of test images has started:")
    y_pred = clf.predict(X_test)
    print("Prediction:", y_pred)

    #accuracy score provides a float between 0 and 1  representing the number of test samples that were correctly classified by the k-nearest neighbour classifier
    print("Accuracy score:", str(accuracy_score(y_test, y_pred)))

    #returns a distance matrix of all pairwise distances between histograms in training samples and test samples
    dist_matrix = distance_matrix(X_test, X_train)


    def calculate_recognition_rate_at_rank_1(distance_matrix, training_set_class_labels, testing_set_class_labels):
        indices = np.argmin(distance_matrix, axis=1)
        closest_class_labels = [training_set_class_labels[index] for index in indices]
        recognition_rate = np.mean(np.array(closest_class_labels) == np.array(testing_set_class_labels))
        return recognition_rate

    def cumulative_match_curve(distance_matrix, training_set_class_labels, testing_set_class_labels):
        indices = np.argsort(distance_matrix, axis=1) #sorts the distance_matrix along the rows, using np.argsort, to obtain the indices of the closest neighbors in the training set for each test sample
        sorted_class_labels = np.array([training_set_class_labels[index] for index in indices]) #2D numpy array of closest class labels using the indices and the training_set_class_labels
        testing_set_class_labels = np.array(testing_set_class_labels)[:, np.newaxis]
        matches_at_every_rank = np.zeros(len(lbp_histograms)) #will increase the indice in the array by 1 when a match is identified at the given column with the same indice
        for i in range(0, len(dist_matrix)):
            correct_class_label = testing_set_class_labels[i]
            sorted_identities = sorted_class_labels[i]
            for rank in range(1, distance_matrix.shape[1] + 1):
                if correct_class_label in sorted_identities[:rank]:
                    matches_at_every_rank[rank-1] += 1
                    break
        return matches_at_every_rank


    def cmc(matches_at_every_rank):
        cumulative_match_scores = []
        for rank in range(1, dist_matrix.shape[1] + 1):
            cumulative_match_scores.append(sum(matches_at_every_rank[:rank]) / len(y_test))
        return cumulative_match_scores

    recog_rate_at_rank_1 = calculate_recognition_rate_at_rank_1(dist_matrix, y_train, y_test)
    matches_at_every_rank = cumulative_match_curve(dist_matrix, y_train, y_test)
    print(matches_at_every_rank)
    cumulative_match_scores = cmc(matches_at_every_rank)
    print(cumulative_match_scores)


    import matplotlib.pyplot as plt

    print(recog_rate_at_rank_1)
    #print(cumulative_match)

    plt.plot([1], [recog_rate_at_rank_1], 'ro')
    #plt.plot(range(1, dist_matrix.shape[1] + 1), cumulative_match, 'g-')
    plt.plot(range(1, dist_matrix.shape[1] + 1), cumulative_match_scores, 'g-')
    plt.xlabel('Rank')
    plt.ylabel('Probability of Identification')
    plt.title('Recognition Rate at Rank 1 and Cumulative Match Curve')
    plt.show()

main()