import cv2
import scipy.misc
import numpy as np
import csv
import os
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


DATA_DIR = 'J:/Data-sets/output/Ch2_002'
INPUT_CSV = '/interpolated_center.CSV'
OUTPUT_DIR= 'resized'

images = []
angles = []
with open(DATA_DIR + INPUT_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename']
        filename1 = row ['angle']
        images.append(filename)
        angles.append(filename1)

'''
length=len(images)
#for i, filename in enumerate(images):
for i in range (0,length):
    ofimages=str('flow_7_local'+ '/' + str(i) +'.jpg')
    images.append(ofimages)
    angles.append(angles[i])

'''
#print(len(images))
#print(len(angles))

c = list(zip(images, angles))
random.shuffle(c)
X_train, X_validation = train_test_split(c, test_size=0.2)
X_train1=X_train
X_validation1=X_validation

num_images=len(X_train1)


def prepare(data,batch_size):
    x,y=zip(*data)
    imgout=[]
    angout=[]
    for i in range(0,batch_size):
        img = scipy.misc.imread(DATA_DIR + '/' + x[i], mode='RGB')
        imgc = scipy.misc.imresize(img[-400:], [66, 200])/255.0
        imgout.append(imgc)
        angout.append((float(y[i])+3.0)*100.0)
    return imgout,angout
    
        
def batch_gen(data,batch_size):
    global X_train
    global X_validation
    batch_imgs=[]
    batch_angles=[]
    batch_imgs,batch_angles=prepare(data,batch_size)
    if (len(data)==len(X_train)):
        X_train = X_train[((int)(batch_size)):len(X_train)]
        if (len(X_train)<((int)(batch_size))):
            X_train=X_train
            print(len(X_train))
            X_train=X_train1
        print(len(X_train))
    elif(len(data)==len(X_validation)):
        X_validation = X_validation[((int)(batch_size)):len(X_validation)]
        if (len(X_validation)<((int)(batch_size))):
            X_validation = X_validation
            print(len(X_validation))
            X_validation = X_validation1
        print(len(X_validation))
    return batch_imgs, batch_angles
