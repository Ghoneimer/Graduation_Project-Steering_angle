import cv2
import scipy.misc
import numpy as np
import csv
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        
#print("Loaded {} samples from file {}".format(len(images),DATA_DIR+INPUT_CSV))

c = list(zip(images, angles))
random.shuffle(c)
X_train, X_validation = train_test_split(c, test_size=0.2)
X_train1=X_train
X_validation1=X_validation
num_images=len(X_train1)

# Flip image horizontally, flipping the angle positive/negative
def horizontal_flip(image, steering_angle):
    flipped_image = cv2.flip(image,1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle
 
# Shift width/height of the image by a small fraction of the total value, introducing an small angle change
def height_width_shift(image, steering_angle, width_shift_range=50.0, height_shift_range=5.0):
    # translation
    tx = width_shift_range * np.random.uniform() - width_shift_range / 2
    ty = height_shift_range * np.random.uniform() - height_shift_range / 2
 
    # new steering angle
    steering_angle += tx / width_shift_range * 2 * 0.2 
 
    transform_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    rows, cols, channels = image.shape
 
    translated_image = cv2.warpAffine(image, transform_matrix, (cols, rows))
    return translated_image, steering_angle
 
# Increase the brightness by a certain value or randomly
def brightness_shift(image, bright_increase=None):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
 
    if bright_increase:
        image_hsv[:,:,2] += bright_increase
    else:
        bright_increase = int(30 * np.random.uniform(-0.3,1))
        image_hsv[:,:,2] = image[:,:,2] + bright_increase
 
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    return image
 
# Rotate the image randomly up to a range_degrees
def rotation(image, range_degrees=5.0):
    degrees = np.random.uniform(-range_degrees, range_degrees)
    rows,cols = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((cols/2,rows/2),degrees,1.0)
    image = cv2.warpAffine(image, matrix, (cols,rows), borderMode=cv2.BORDER_REPLICATE)
    return image
 
# Zoom the image randomly up to zoom_range, where 1.0 means no zoom and 1.2 a 20% zoom
def zoom(image, zoom_range=(1.0,1.2)): 
    # resize
    factor = np.random.uniform(zoom_range[0], zoom_range[1])
    height, width = image.shape[:2]
    new_height, new_width = int(height*factor), int(width*factor)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
 
    # crop margins to match the initial size
    start_row = int((new_height-height)/2)
    start_col = int((new_width-width)/2)
    image = image[start_row:start_row + height, start_col:start_col + width]
 
    return image

# Crop and resize the image
def resize_image(image):
    # resize to the final sizes even the aspect ratio is destroyed
    image = scipy.misc.imresize(image[-400:], [66, 200])    
    return image


#transform for data
def transform(image,steering):
    if np.random.random() < 0.5: 
        tran_image, tran_steering = horizontal_flip(image, steering)
    tran_image, tran_steering = height_width_shift(image, steering, width_shift_range=50.0, height_shift_range=5.0)
    tran_image = zoom(image, zoom_range=(1.0,1.2))
    #tran_image = rotation(image, range_degrees=5.0)
    #tran_image = brightness_shift(image, bright_increase=None)
    tran_image=tran_image/255.0 #normalization
    return tran_image, tran_steering

def load_and_augment(data):
    #read the image 
    image = scipy.misc.imread(DATA_DIR + '/' + data[0], mode='RGB')
    steering = float(data[1])
    #resize the image to the standard width and height size for the CNN
    image = resize_image(image)
    #make the random operations
    image2, steering2 = transform(image, steering)
    return image,image2, steering, steering2

#this function generate the batches for the validation and training test

def batch_gen (data, batch_size, height=66, width=200):
    global X_train
    global X_validation
    #initialize the batches lists
    batch_images = np.zeros((batch_size, height, width, 3))
    batch_steering_angles = np.zeros(batch_size)
    for i in range ((int)(batch_size/2)):
        image, image2, steering, steering2 = load_and_augment(data[i])
        batch_images[i] = image
        m = (int)(i+(batch_size/2))
        batch_images[m] = image2
        batch_steering_angles[i] = steering
        batch_steering_angles[m] = steering2
    if (len(data)==len(X_train)):
        X_train = X_train[((int)(batch_size/2)):len(X_train)]
        print(len(X_train))
        if (len(X_train)<((int)(batch_size))):
            X_train=X_train
            print(len(X_train))
            X_train=X_train1
    elif(len(data)==len(X_validation)):
        X_validation = X_validation[((int)(batch_size/2)):len(X_validation)]
        print(len(X_validation))
        if (len(X_validation)<((int)(batch_size))):
            X_validation = X_validation
            print(len(X_validation))
            X_validation = X_validation1
    '''
    #save the images in a folder for debug purposes only
    for j in range (len(batch_images)):
        cv2.imwrite(data_path+'batch/'+str(j)+".jpg", batch_images[j])
    #those next two lines are for debugging only
    print(len(batch_images))
    print(i)
    '''
    return batch_images, batch_steering_angles
    
#just run this function into your code and you will get a list of 
#batch_images, batch_steering_angles
#at training time path the X_train dataset to it 
#and at validation time use the same function but path the X_validation dataset to it 
#batch = batch_gen(X_train,32)
#print(batch[1])
'''
#plot the histogram to explort the data
print("\nExploring the dataset after augmentation ...")
 
# It plots the histogram of an arrray of angles: [0.0,0.1, ..., -0.1]
def plot_steering_histogram(steerings, title, num_bins=100):
    plt.hist(steerings, num_bins)
    plt.title(title)
    plt.xlabel('Steering Angles')
    plt.ylabel('# Images')
    plt.show()
  
# Plot the histogram of steering angles after the image augmentation
plot_steering_histogram(batch[1], 'Number of images per steering angle before image augmentation', num_bins=100)
print("Exploring the dataset complete.")
'''
