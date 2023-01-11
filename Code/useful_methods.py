import os

from imutils import paths

class UsefulMethods:

    def getDataset(path):

        img_data_array = []
        class_name = []

        for dir1 in os.listdir(path):
            for file in os.listdir(os.path.join(path, dir1)):
                image_path = os.path.join(path, dir1, file)
