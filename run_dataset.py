import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import scipy.misc
import model
import cv2
import csv
#from subprocess import call

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess,"D:/FOE/Graduation project/Models/Autopilot-TensorFlow-master/save/model2/modelsteering")

img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = img.shape

smoothed_angle = 0

DATA_DIR = 'J:/Data-sets/output/Ch2_002'
INPUT_CSV = '/interpolated_center.CSV'

images=[]
#angles=[]
with open(DATA_DIR + INPUT_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['filename']
        filename1 = row['angle']
        images.append(filename)
        #angles.append(float(filename1))

i = 0
while(cv2.waitKey(10) != ord('q')):
    full_image = scipy.misc.imread(DATA_DIR + '/' + images[i], mode="RGB")
    image = scipy.misc.imresize(full_image[-400:], [66, 200]) / 255.0
    degrees = model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180.0 / scipy.pi
    #degrees = angles[i]*180.0/scipy.pi
    #call("clear")
    print("Predicted steering angle: " + str(degrees) + " degrees")
    cv2.imshow("frame", cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
    #make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    #and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),smoothed_angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i += 1

cv2.destroyAllWindows()
