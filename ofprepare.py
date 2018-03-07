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
num_images1=len(images)

b = list(zip(images, angles))
c=[]
#random.shuffle(c)
for i in range(0,int(num_images1/10)):
    c.append(b[10*i:10*i+10])
e=b[int(num_images1/10)*10:]
e=np.reshape(e,np.shape(e))
np.random.shuffle(c)
X_train, X_validation = train_test_split(c, test_size=0.2)
sh1=np.shape(X_train)
sh2=np.shape(X_validation)
X_train=np.reshape(X_train,(sh1[0]*sh1[1],sh1[2]))
X_validation=np.reshape(X_validation,(sh2[0]*sh2[1],sh2[2]))
X_train=np.append(X_train,e)
X_train=np.reshape(X_train,(int(len(X_train)/2),2))
X_train1=X_train
X_validation1=X_validation

num_images=len(X_train1)

def prepare(data,batch_size):
    imgout=[]
    angout=[]
    for i in range(0,batch_size):
        img = scipy.misc.imread(DATA_DIR + '/' + data[i][0], mode='RGB')
        imgc = scipy.misc.imresize(img[-400:], [66, 200])/255.0
        #cv2.imwrite(DATA_DIR + '/' + OUTPUT_DIR + '/' + str(i) + '.jpg' , imgc)
        imgout.append(imgc)
        angout.append((float(data[i][1])+3.0)*100.0)
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
    
#batch_gen(X_train,10)
