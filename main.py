'''
author: Andreas Askim Vatne
version: 1.0
date: 11/01/2023
'''

#imports lbp descriptor code from the 'Code' module
from Code import lbp 

#imports machine learning library SVM for classification
'''
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection
Source: https://scikit-learn.org/stable/modules/svm.html#svm-classification
'''
from sklearn import svm

#other important utilites and libraries
from imutils import paths
import argparse
import cv2
import os

def main():

    #parses arguments provided by user in command line
    ap = argparse.ArgumentParser(description="Program runs the LBP algorithm and recognizes classes based on face images from database: ") 
    ap.add_argument("-t", "--training", required=True, help="path to training images")
    ap.add_argument("-e", "--testing", required=True, help="path to testing images")

    arguments = vars(ap.parse_args())

    #creates an instance of the LBP decriptor
    lbp_descriptor = lbp.LBP(img, 1, 8)