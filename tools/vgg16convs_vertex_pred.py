import glob
import os

from math import ceil
import numpy as np
from scipy.io import loadmat

import cv2

from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add
from keras.layers import Concatenate, Dropout, Dense
from keras.models import Model

from keras import backend as K
from keras.layers import Layer
import keras.layers as KL
import keras.engine as KE


############################################################
#  Network Class
############################################################

class vgg16convs_vertex_pred():

    def __init__(self, input):

        self.build(input)

    def build(self, input):

        # 1st block
        conv1_1 = Conv2D(64, (3,3), name='conv1_1', padding='same', activation='relu')(input)
        conv1_2 = Conv2D(64, (3,3), name='conv1_2', padding='same', activation='relu')(conv1_1)
        pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_2)

        # 2nd block
        conv2_1 = Conv2D(128, (3,3), name='conv2_1', padding='same', activation='relu')(pool1)
        conv2_2 = Conv2D(128, (3,3), name='conv2_2', padding='same', activation='relu')(conv2_1)
        pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)

        # 3rd block
        conv3_1 = Conv2D(256, (3,3), name='conv3_1', padding='same', activation='relu')(pool2)
        conv3_2 = Conv2D(256, (3,3), name='conv3_2', padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(256, (3,3), name='conv3_3', padding='same', activation='relu')(conv3_2)
        pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_3)

        # 4th block
        conv4_1 = Conv2D(512, (3,3), name='conv4_1', padding='same', activation='relu')(pool3)
        conv4_2 = Conv2D(512, (3,3), name='conv4_2', padding='same', activation='relu')(conv4_1)
        conv4_3 = Conv2D(512, (3,3), name='conv4_3', padding='same', activation='relu')(conv4_2)
        pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_3)

        # 5th block
        conv5_1 = Conv2D(512, (3,3), name='conv5_1', padding='same', activation='relu')(pool4)
        conv5_2 = Conv2D(512, (3,3), name='conv5_2', padding='same', activation='relu')(conv5_1)
        conv5_3 = Conv2D(512, (3,3), name='conv5_3', padding='same', activation='relu')(conv5_2)

        # vertex pred head
        num_units = 64
        keep_prob_queue = 0.5 # YX uses keep_prob
        rate = 1 - keep_prob_queue # keras uses rate
        scale = 1.

        score_conv5 = Conv2D(num_units, (1,1), name='score_conv5', padding='same', activation='relu')(conv5_3)
        # upscore_conv5 = deconv(score_conv5, 4, 4, num_units, 2, 2, name='upscore_conv5', trainable=False)  # how to make a keras equivalent layer?
        upscore_conv5 = KL.Conv2DTranspose(num_units, (4,4), strides=(2,2), name='upscore_conv5', padding='same', data_format="channels_last", trainable=False)(score_conv5)
        print(upscore_conv5.shape)

        score_conv4 = Conv2D(num_units, (1,1), name='score_conv4', padding='same', activation='relu')(conv4_3)

        add_score = Add()([score_conv4, upscore_conv5])
        dropout = Dropout(rate, name='dropout')(add_score)
        #upscore = deconv(dropout, int(16*scale), int(16*scale), num_units, int(8*scale), int(8*scale), name='upscore', trainable=False)
        upscore = KL.Conv2DTranspose(num_units, (int(16*scale), int(16*scale)), strides=(int(8*scale), int(8*scale)), name='upscore', padding='same', data_format="channels_last", trainable=False)(dropout)


        # 'prob' and 'label_2d' will be added later. 'gt_label_weight' cannot be added because of hard_label C++ module

        vertex_reg = 1
        num_classes = 3
        if vertex_reg:
            score_conv5_vertex = Conv2D(128, (1,1), name='score_conv5_vertex', padding='same', activation='relu')(conv5_3)
            #   upscore_conv5_vertex = deconv(score_conv5_vertex, 4, 4, 128, 2, 2, name='upscore_conv5_vertex')
            upscore_conv5_vertex = KL.Conv2DTranspose(128, (4, 4), strides=(2, 2), name='upscore_conv5_vertex', padding='same', data_format="channels_last", trainable=False)(score_conv5_vertex)

            
            score_conv4_vertex = Conv2D(128, (1,1), name='score_conv4_vertex', padding='same', activation='relu')(conv4_3)
            
            add_score_vertex = Add()([score_conv4_vertex, upscore_conv5_vertex])
            dropout_vertex = Dropout(rate, name='dropout_vertex')(add_score_vertex)
            #   upscore_vertex = deconv(dropout_vertex, int(16*scale), int(16*scale), 128, int(8*scale), int(8*scale), name='upscore_vertex', trainable=False)
            upscore_vertex = KL.Conv2DTranspose(128, (int(16*scale), int(16*scale)), strides=(int(8*scale), int(8*scale)), name='upscore_vertex', padding='same', data_format="channels_last", trainable=False)(dropout_vertex)
            
            # 3*num_classes -> a fixed output like this 
            vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu')(upscore_vertex)


        # create keras model
        self.the_model = Model(inputs=input, outputs=vertex_pred)


############################################################
#  Data Generator
############################################################

def prepare_dataset_indexes(data_path):
    database = []
    num_color = len(glob.glob(data_path + '/*-color.png'))
    dat_size = num_color * 4
    for i in range(dat_size):
        data = {}
        data['color'] = os.path.join(data_path, '{:06}-color.png'.format(i))
        data['meta'] = os.path.join(data_path, '{:06}-meta.mat'.format(i))
        database.append(data)

    return database


def data_generator(data_path=None, shuffle=True, batch_size=1):

    b = 0
    index = -1
    dataset_indexes = prepare_dataset_indexes(data_path)

    print len(dataset_indexes)
    while True:
        index = (index + 1) % len(dataset_indexes)
        if shuffle and index == 0:
            np.random.shuffle(dataset_indexes)

        data_rec = dataset_indexes[index]

        rgb = imread(data_rec["color"])
        mat = loadmat(data_rec["meta"])

        


    
############################################################
#  Main
############################################################

if __name__ == "__main__":
    # execute only if run as a script
    # main()

    input = Input(shape=(480, 640, 3))
    print(input.shape)

    mdw = vgg16convs_vertex_pred(input)
    print(mdw.the_model.output_shape)

    mdw.the_model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    mdw.the_model.summary()              

    data_path = '/home/shawnle/Documents/Restore_PoseCNN/PoseCNN-master/data_syn_LOV/data_2_objs/'
    dat_gen = data_generator(data_path)