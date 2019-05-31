import tensorflow as tf
import numpy as np
import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_utils import load_image

SCRIPT_PATH = os.path.dirname(os.path.abspath( __file__ ))
num = 548
imgs = []
for i in range(1, num + 1):
    imgs.append(np.asarray(load_image("%s/endo/%s.tif" % (SCRIPT_PATH, i))))
for i in range(1, num + 1):
    imgs.append(np.asarray(load_image("%s/noen/%s.tif" % (SCRIPT_PATH, i))))
imgs = np.array(imgs)
y_data = np.r_[np.c_[np.ones(num), np.zeros(num)],np.c_[np.zeros(num), np.ones(num)]]
print (imgs.shape)
print (y_data.shape)

x_test = []
for i in range(1, 11):
    x_test.append(np.asarray(load_image("%s/TestSet/%s.tif" % (SCRIPT_PATH, i))))
x_test =  np.array(x_test)
y_test = np.r_[np.c_[np.ones(5), np.zeros(5)],np.c_[np.zeros(5), np.ones(5)]]
print (x_test.shape)
print (y_test.shape)

# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 4], name='input')
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001, name='target')


# Training

model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit({'input': imgs}, {'target': y_data}, n_epoch=20,
           validation_set=({'input': x_test}, {'target': y_test}),
           snapshot_step=100,show_metric=True, run_id='AlexNet_DPR')
model.save('DPR_AlexNet_model.tflearn')