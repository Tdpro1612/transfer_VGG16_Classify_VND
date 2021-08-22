import pandas as pd
import numpy as np
import cv2 as cv2
import pickle
from matplotlib import pyplot
import os
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.image as image

#test thu anh

#img = cv2.imread("D:\\anh_test.jpg")

#cv2.imshow("img", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



class_name = ["00000","1000","2000","5000","10000","20000","50000","100000","200000,"500000"]

raw_folder = "D:\\VND-pics\\"
def save_data(raw_folder=raw_folder):

    dest_size = (224, 224)
    print("Bắt đầu xử lý ảnh...")

    pixels = []
    labels = []

    # Lặp qua các folder con trong thư mục raw
    for folder in listdir(raw_folder):
        if folder!='.DS_Store':
            print("Folder=",folder)
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(raw_folder  + folder):
                if file!='.DS_Store':
                    #print("File=", file)
                    #print(raw_folder  + folder +"\\" + file)
                    path = raw_folder  + folder +"\\" + file
                    img = cv2.imread(path)
                    img = cv2.resize(img,dsize=(224,224))
                    pixels.append(img)
                    labels.append(folder)

    pixels = np.array(pixels)
    labels = np.array(labels)#.reshape(-1,1)

    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels)
    print(labels)

    file = open('pix.data', 'wb')
    # dump information to that file
    pickle.dump((pixels,labels), file)
    # close the file
    file.close()

    return
save_data(raw_folder)
