import hashlib
import os
import pickle
from urllib.request import urlretrieve
import random

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample

from zipfile import ZipFile

print('All modules imported.')

import tensorflow as tf

def download(url, file):
    """
    Download file from <url>
    :param url: URL to file
    :param file: Local file path
    """
    if not os.path.isfile(file):
        print('Downloading ' + file + '...')
        urlretrieve(url, file)
        print('Download Finished')

download('https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip', 'traffic-signs-data.zip')
print("downloaded")

def uncompress_files(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    with ZipFile(file) as zipf:
        zipf.extractall('traindata')
uncompress_files('traffic-signs-data.zip')
os.listdir('traindata')

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'traindata/train.p'
validation_file='traindata/valid.p'
testing_file = 'traindata/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


class_count = {}

for label in y_train.tolist():
    class_count[label] = class_count.get(label, 0) + 1


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
import cv2
def addHsv(x):
    return np.array([np.dstack((i,cv2.cvtColor(i,cv2.COLOR_RGB2HSV))) for i in x])

def toHsv(x):
    return np.array([cv2.cvtColor(i,cv2.COLOR_RGB2HSV) for i in x])

def translateImg(x,pX,pY):
    M = np.float32([[1,0,x.shape[0]*pX],[0,1,x.shape[0]*pY]])
    res = cv2.warpAffine(x,M,(x.shape[0],x.shape[1]))
    return res

def rotateImg(x,theta):
    cols = x.shape[0]
    rows = x.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),theta,1)#random.uniform(0.999,1.001))
    res = cv2.warpAffine(x,M,(cols,rows))
    return res

def normalize_img(image_data):
    return (image_data / 128) - 1

aug_img = []
aug_lbl = []
for _ in range(3):
    for x,l in zip(X_train, y_train):
        if class_count[l] < 1500:
            aug_img.append(translateImg(x,random.uniform(-0.05,0.05),random.uniform(-0.05,0.05)))
            aug_img.append(rotateImg(translateImg(x,random.uniform(-0.05,0.05),random.uniform(-0.05,0.05)),random.uniform(-8,8)))
            aug_img.append(rotateImg(x,random.uniform(-8,8)))
            aug_lbl.append(l)
            aug_lbl.append(l)
            aug_lbl.append(l)
            class_count[l]+=3
    
for x,l in zip(X_train, y_train):
    aug_img.append(x)
    aug_lbl.append(l)

X_train = np.array(aug_img)
y_train = np.array(aug_lbl)
aug_img = None
aug_lbl = None
# X_train = toHsv(X_train)
# X_valid = toHsv(X_valid)
# X_test = toHsv(X_test)

X_train = normalize_img(X_train)
X_valid = normalize_img(X_valid)
X_test = normalize_img(X_test)
print(max(X_train[0].flatten()),min(X_train[0].flatten()))


### Define your architecture here.
### Feel free to use as many code cells as needed.
import random
import numpy as np

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCHS = 50
BATCH_SIZE = 128
KEEP_PROB = 0.55


from tensorflow.contrib.layers import flatten
def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    #Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x3.
    FilterWeight1p = tf.Variable(tf.truncated_normal((1,1,3,3),mu,sigma))
    FilterBias1p = tf.Variable(tf.zeros(3))
    strides=[1,1,1,1]
    padding="VALID"
    conv1p = tf.nn.conv2d(x, FilterWeight1p, strides, padding)
    conv1p = tf.nn.bias_add(conv1p, FilterBias1p)
    
    # TODO: Activation.
    actv1 = tf.nn.relu(conv1p)
    
    # TODO: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    FilterWeight1 = tf.Variable(tf.truncated_normal((5,5,3,6),mu,sigma))
    FilterBias1 = tf.Variable(tf.zeros(6))
    strides=[1,1,1,1]
    padding="VALID"
    conv1 = tf.nn.conv2d(actv1, FilterWeight1, strides, padding)
    conv1 = tf.nn.bias_add(conv1, FilterBias1)
    
    # TODO: Activation.
    actv1 = tf.nn.relu(conv1)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    pool1 = tf.nn.max_pool(actv1, ksize, strides, padding)
    
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    FilterWeight2 = tf.Variable(tf.truncated_normal((5,5,6,16),mu,sigma))
    FilterBias2 = tf.Variable(tf.zeros(16))
    strides = [1,1,1,1]
    padding = "VALID"
    conv2 = tf.nn.conv2d(pool1, FilterWeight2, strides, padding)
    conv2 = tf.nn.bias_add(conv2, FilterBias2)
    
    # TODO: Activation.
    actv2 = tf.nn.relu(conv2)
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    padding = "VALID"
    pool2 = tf.nn.max_pool(actv2, ksize, strides, padding)
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat_data = tf.contrib.layers.flatten(pool2)
    #print(flat_data)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    w1 = tf.Variable(tf.truncated_normal((400, 120),mu,sigma))
    b1 = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(flat_data,w1),b1)
    
    # TODO: Activation.
    fc1_act = tf.nn.relu(fc1)
    
    fc1_act = tf.nn.dropout(fc1_act, keep_prob)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    w2 = tf.Variable(tf.truncated_normal((120, 84),mu,sigma))
    b2 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1_act,w2),b2)
    
    # TODO: Activation.
    fc2_act = tf.nn.relu(fc2)
    
    fc2_act = tf.nn.dropout(fc2_act, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    w3 = tf.Variable(tf.truncated_normal((84, n_classes),mu,sigma))
    b3 = tf.Variable(tf.zeros(n_classes))
    fc3 = tf.add(tf.matmul(fc2_act,w3),b3)

    logits = fc3
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation,loss_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return (total_accuracy / num_examples, total_loss / num_examples)


    ### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    best_train_accuracy = 0
    best_valid_accuracy = 0
    print("Training...")
    print()
    loss_history = {"train":[],"valid":[]} 
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: KEEP_PROB})
            
        (validation_accuracy, validation_loss) = evaluate(X_valid, y_valid)
        (training_accuracy, training_loss) = evaluate(X_train, y_train)
        
        loss_history["valid"].append(validation_loss)
        loss_history["train"].append(training_loss)
        
        print("EPOCH {} ...".format(i+1))
        #print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        if training_accuracy>=best_train_accuracy and validation_accuracy>=best_valid_accuracy:
            best_train_accuracy = training_accuracy
            best_valid_accuracy = validation_accuracy
            saver.save(sess, './best-lenet')
        
    saver.save(sess, './lenet')
    print("Model saved")



with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy,test_loss = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

with tf.Session() as sess:
    saver.restore(sess, "./best-lenet")
    test_accuracy,test_loss = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
