'''
project:    LBP-PROJECT
author:     Andreas Askim Vatne
version:    1.0
date:       12-02-2023
github:     https://github.com/aav-98/LBP-PROJECT
'''

#import os to be able to traverse directories and files in operating system
import os
from matplotlib import pyplot as plt

#import pandas to organize and extract information from datasets
import pandas as pd

from PIL import Image
import numpy as np
import cv2     #open-source computer vision library

def getDataset(path_segment):   #returns the dataset partitioned into relevant arrays and placed into dataframe

    class_labels = []
    image_paths = []
    image_labels = []

    dataset_path = os.getcwd() + "/" + path_segment + "/"   #creates a path to the required directory containing requested dataset images

    haar_cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for class_directory in os.listdir(dataset_path):
        class_images = os.listdir(os.path.join(dataset_path, class_directory))
        for image in class_images:
            image_path = os.path.join(dataset_path, class_directory, image)
            faces_rect, image_mat = detect_faces(haar_cascade_face, image_path)
            if len(faces_rect)!=1:
               continue  
            cropped_image, new_image_path = crop(image_mat, faces_rect, image_path)
            image_paths.append(new_image_path)
            image_labels.append(image)
            class_labels.append(class_directory)

    data = {"image_paths":image_paths, "image_labels":image_labels, "class_labels":class_labels}
    imageseries = pd.DataFrame(data)    #create a dataframe from the data extracted above

    return imageseries

#save lbp-images created and stored in dataframe
def saveLBPImages(imageSeries:pd.DataFrame, path_segment):
    for index in imageSeries.index:
        lbp_image = imageSeries["lbp_images"][index]
        image_label = imageSeries["image_labels"][index]
        path = os.getcwd() + "/" + path_segment + "/" + image_label + ".jpg"
        img = Image.fromarray(lbp_image.astype("uint8"), 'L')
        img.save(path)

#split the lbp image into tiles
def split_lbp_image(lbp_image, height, width):
    #image size is 168 * 192
    M = lbp_image.shape[0]//height #shape[0] gives image height
    N = lbp_image.shape[1]//width  #shape[1] gives image width
    tiles = [lbp_image[x:x+M,y:y+N] for x in range(0,lbp_image.shape[0],M) for y in range(0,lbp_image.shape[1],N)]
    return tiles

#calculate the recogntion rate
def calculate_recognition_rate_at_rank_1(distance_matrix, training_set_class_labels, testing_set_class_labels):
    #find indices of the closest neighbors in training set for each test sample by finding minimum distance in the distance_matrix along the rows
    indices = np.argmin(distance_matrix, axis=1) 
    closest_class_labels = [training_set_class_labels[index] for index in indices] #creates a list of closest class labels using the indices
    recognition_rate = np.mean(np.array(closest_class_labels) == np.array(testing_set_class_labels)) #result of calculation represents the recognition rate at rank 1
    return recognition_rate

#calculate the correct matches at each rank, stopping for every row when a correct match is identified
def match_scores(distance_matrix, training_set_class_labels, testing_set_class_labels):
    indices = np.argsort(distance_matrix, axis=1) #sorts the distance_matrix along the rows, using np.argsort, to obtain the indices of the closest neighbors in the training set for each test sample
    sorted_class_labels = np.array([training_set_class_labels[index] for index in indices]) #2D numpy array of sorted class labels using the indices and the training_set_class_labels
    testing_set_class_labels = np.array(testing_set_class_labels)[:, np.newaxis] #converts the testing set class labels into similar format at sorted class labels
    matches_at_every_rank = np.zeros(len(training_set_class_labels)) #will increase the indice in the array by 1 when a match is identified at the given column with the same indice
    for i in range(0, len(testing_set_class_labels)):
        correct_class_label = testing_set_class_labels[i]
        sorted_identities = sorted_class_labels[i]
        for rank in range(1, distance_matrix.shape[1] + 1):
            if correct_class_label in sorted_identities[:rank]:
                matches_at_every_rank[rank-1] += 1
                break
    return matches_at_every_rank

#calculate the cumulative match curve scores to plot cmc
def cumulative_match_curve_scores(matches_at_every_rank, nr_instances):
    cumulative_match_curve_scores = []
    for rank in range(1, len(matches_at_every_rank) + 1):
        cumulative_match_curve_scores.append(sum(matches_at_every_rank[:rank]) / nr_instances) #divides the number of correctly identified matches at a rank with the nr of test images
    return cumulative_match_curve_scores

def detect_faces(cascade, image, scaleFactor = 1.1):
    image_mat = cv2.imread(image).copy()

    # Applying the haar classifier to detect faces
    faces = cascade.detectMultiScale(image_mat, scaleFactor=scaleFactor, minNeighbors=5)

    return faces, image_mat
    
def crop(image, faces, image_path, k=0):
    for (x, y, w, h) in faces:
        cropped_image = image[y:y + h, x:x + w]
        # comment the two following lines if you want to stop saving the crops
        cropped_image_normalized = cv2.resize(cropped_image, (168, 192))
        cropped_image_path = image_path+str(k)+'.jpg'
        cv2.imwrite(cropped_image_path, cropped_image_normalized)
        k += 1
    return cropped_image, cropped_image_path