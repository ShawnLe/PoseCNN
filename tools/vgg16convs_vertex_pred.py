from __future__ import print_function

import _init_paths

import glob
import os

from math import ceil
import numpy as np
from scipy.io import loadmat

import cv2

import tensorflow as tf
from tensorflow.python.ops import state_ops

from fcn.config import cfg

# import keras via tensorflow
# keras bug: conflict keras version
# bug message: FailedPreconditionError (see above for traceback): Error while reading resource variable vertex_pred/kernel from Container: localhost. This could mean that the variable was uninitialized. Not found: Container localhost does not exist. (Could not find resource: localhost/vertex_pred/kernel)
# bug linK: https://github.com/horovod/horovod/issues/511
# from tensorflow.python import keras

# from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add
# from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Concatenate, Dropout, Dense, Lambda, Activation
# from tensorflow.python.keras.models import Model, load_model
# from tensorflow.python.keras.activations import relu
# from tensorflow.python.keras.models import load_model

# from keras.layers import Concatenate, Dropout, Dense, Lambda
# from keras.models import Model
# from keras.models import load_model

# from tensorflow.python.keras import backend as K
# import keras.backend as K
# from tensorflow.python.keras.layers import Layer
# import tensorflow.python.keras.layers as KL
# import tensorflow.python.keras.engine as KE
# from keras import backend as K
# from keras.layers import Layer
# import keras.layers as KL
# import keras.engine as KE

from numpy import linalg as LA


############################################################
#  layer library
############################################################

DEFAULT_PADDING = 'SAME'

def make_var(name, shape, initializer=None, regularizer=None, trainable=True):
    return tf.get_variable(name, shape, initializer=initializer, regularizer=regularizer, trainable=trainable)

def validate_padding(padding):
    assert padding in ('SAME', 'VALID')

def max_pool(input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
    validate_padding(padding)
    return tf.nn.max_pool(input,
                            ksize=[1, k_h, k_w, 1],
                            strides=[1, s_h, s_w, 1],
                            padding=padding,
                            name=name)

def conv(input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True, biased=True, c_i=-1):
    validate_padding(padding)
    if isinstance(input, tuple):
        input = input[0]
    if c_i == -1:
        c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name, reuse=reuse) as scope:
        init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
        regularizer = tf.contrib.layers.l2_regularizer(scale=cfg.TRAIN.WEIGHT_REG)
        kernel = make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, regularizer, trainable)
        # print ("kernel ", kernel)
        # print ("imput ", input)
        if group==1:
            output = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
            output = tf.concat(3, output_groups)
        # Add the biases
        if biased:
            init_biases = tf.constant_initializer(0.0)
            biases = make_var('biases', [c_o], init_biases, regularizer, trainable)
            output = tf.nn.bias_add(output, biases)
        if relu:
            output = tf.nn.relu(output, name=scope.name)    
    return output


def load(data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            print op_name
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        print op_name + ' ' + param_name + ' assigned'
                    except ValueError:
                        if not ignore_missing:
                            raise
            # try to assign dual weights
            with tf.variable_scope(op_name+'_p', reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        print op_name + '_p ' + param_name + ' assigned'
                    except ValueError:
                        if not ignore_missing:
                            raise

            with tf.variable_scope(op_name+'_d', reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

############################################################
#  Network Class
############################################################

class vgg16convs_vertex_pred():

    # def __init__(self, input):
    def __init__(self, shape=(None,None,None), trainable=True):

        # self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.input = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2]])
        self.trainable = trainable

        self.build()

    def backend_debug_print(self, x):
        K.print_tensor(x, message='hello_print') # , [tf.shape(x)]
        return x

    def smooth_l1_loss_vertex(self, vertex_targets_, vertex_pred):

        sigma = 1.0

        vertex_targets, vertex_weights = vertex_targets_[...,:9],  vertex_targets_[...,9:] 

        sigma_2 = sigma ** 2
        vertex_diff = vertex_pred - vertex_targets
        diff = tf.multiply(vertex_weights, vertex_diff)
        abs_diff = tf.abs(diff)
        # tf.less (https://stackoverflow.com/questions/44759779/comparison-with-tensors-python-tensorflow)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_diff, 1. / sigma_2)))
        in_loss = tf.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        loss = tf.div( tf.reduce_sum(in_loss), tf.reduce_sum(vertex_weights) + 1e-10 )
        return loss

    def build(self):

        conv1_1 = conv(self.input, 3, 3, 64, 1, 1, name='conv1_1', c_i=3, trainable=self.trainable)
        conv1_2 = conv(conv1_1, 3, 3, 64, 1, 1, name='conv1_2', c_i=64, trainable=self.trainable)
        pool1 = max_pool(conv1_2, 2, 2, 2, 2, name='pool1')

        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=self.trainable)
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=self.trainable)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')

        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=self.trainable)
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=self.trainable)
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=self.trainable)
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')

        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=self.trainable)
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=self.trainable)
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=self.trainable)
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')

        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=self.trainable)
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=self.trainable)
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=self.trainable)

        return None


        # vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input)
        # print('vgg output :', vgg16.output.get_shape())
        # # until the last 2 layers, all are freezed for training
        # for layer in vgg16.layers:
        #   layer.trainable = True

        # conv4_3 = vgg16.layers[-6].output
        # conv5_3 = vgg16.layers[-2].output # 2nd layer from the last, block5_conv3

        # # vertex pred head
        # num_units = 64
        # keep_prob_queue = 0.5 # YX uses keep_prob
        # rate = 1 - keep_prob_queue # keras uses rate
        # scale = 1.

        # score_conv5 = Conv2D(num_units, (1,1), name='score_conv5', padding='same', activation='relu')(conv5_3)
        # # upscore_conv5 = deconv(score_conv5, 4, 4, num_units, 2, 2, name='upscore_conv5', trainable=False)  # how to make a keras equivalent layer?
        # # Keras bug: before and after KL.Conv2DTranspose, shape is lost. Use tensorflow keras instead. Link: https://github.com/keras-team/keras/issues/6777
        # # score_conv5: (?, 30, 40, 64)
        # # upscore_conv5: (?, ?, ?, 64)
        # print('score_conv5:', score_conv5.shape)
        # upscore_conv5 = Conv2DTranspose(num_units, (4,4), strides=(2,2), name='upscore_conv5', padding='same', data_format="channels_last", trainable=False)(score_conv5)
        # print('upscore_conv5:', upscore_conv5.shape)

        # score_conv4 = Conv2D(num_units, (1,1), name='score_conv4', padding='same', activation='relu')(conv4_3)

        # # score_conv4_p = Lambda(self.backend_debug_print)(score_conv4)  # , output_shape=K.shape(score_conv4)
        # score_conv4_p = K.print_tensor(score_conv4, "checking if data is being fed")

        # # add_score = Add()([score_conv4, upscore_conv5])
        # add_score = Add()([score_conv4_p, upscore_conv5])

        # dropout = Dropout(rate, name='dropout')(add_score)
        
        # #upscore = deconv(dropout, int(16*scale), int(16*scale), num_units, int(8*scale), int(8*scale), name='upscore', trainable=False)
        # upscore = Conv2DTranspose(num_units, (int(16*scale), int(16*scale)), strides=(int(8*scale), int(8*scale)), name='upscore', padding='same', data_format="channels_last", trainable=False)(dropout)
        # print('output shape: ', upscore.get_shape())

        # # 'prob' and 'label_2d' will be added later. 'gt_label_weight' cannot be added because of hard_label C++ module

        # vertex_reg = 1
        # num_classes = 3
        # if vertex_reg:
        #     init_weights = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001)
        #     init_bias = keras.initializers.Constant(value=0.)
        #     regularizer = keras.regularizers.l2(0.0001)

        #     # score_conv5_vertex = Conv2D(128, (1,1), name='score_conv5_vertex', padding='same', activation='relu')(conv5_3)
        #     score_conv5_vertex = Conv2D(128, (1,1), name='score_conv5_vertex', padding='same', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(conv5_3)
        #     #   upscore_conv5_vertex = deconv(score_conv5_vertex, 4, 4, 128, 2, 2, name='upscore_conv5_vertex')
        #     upscore_conv5_vertex = Conv2DTranspose(128, (4, 4), strides=(2, 2), name='upscore_conv5_vertex', padding='same', data_format="channels_last", trainable=False)(score_conv5_vertex)

            
        #     # score_conv4_vertex = Conv2D(128, (1,1), name='score_conv4_vertex', padding='same', activation='relu')(conv4_3)
        #     score_conv4_vertex = Conv2D(128, (1,1), name='score_conv4_vertex', padding='same', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(conv4_3)
            
        #     add_score_vertex = Add()([score_conv4_vertex, upscore_conv5_vertex])
        #     dropout_vertex = Dropout(rate, name='dropout_vertex')(add_score_vertex)
        #     #   upscore_vertex = deconv(dropout_vertex, int(16*scale), int(16*scale), 128, int(8*scale), int(8*scale), name='upscore_vertex', trainable=False)
        #     upscore_vertex = Conv2DTranspose(128, (int(16*scale), int(16*scale)), strides=(int(8*scale), int(8*scale)), name='upscore_vertex', padding='same', data_format="channels_last", trainable=False)(dropout_vertex)
            
        #     # 3*num_classes == depth == # channels-> a fixed output like this 
        #     # vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu')(upscore_vertex)
        #     # vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(upscore_vertex)
        #     vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(upscore_vertex)
        #     # vertex_pred_after = Activation('relu')(vertex_pred_before)
        #     # vertex_pred_after = relu(vertex_pred_before)
            
        #     # vertex_pred = Conv2D(2*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu')(upscore_vertex)
        #     # vertex_pred = K.reshape(vertex_pred, (480, 640, 3*num_classes))
        # # create keras model
        # self.the_model = Model(inputs=input, outputs=vertex_pred)
        # # self.the_model = Model(inputs=input, outputs=vertex_pred_after)
        # # self.__dict__.update(locals())


############################################################
#  Data Generator
############################################################

def pad_im(im, factor, value=0):
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return np.lib.pad(im, ((0, pad_height), (0, pad_width), (0,0)), 'constant', constant_values=value)
    elif len(im.shape) == 2:
        return np.lib.pad(im, ((0, pad_height), (0, pad_width)), 'constant', constant_values=value)


def prepare_dataset_indexes(data_path):
    database = []
    num_color = len(glob.glob(data_path + '/*-color.png'))
    dat_size = num_color 
    for i in range(dat_size):
        data = {}
        data['color'] = os.path.join(data_path, '{:06}-color.png'.format(i))
        data['meta'] = os.path.join(data_path, '{:06}-meta.mat'.format(i))
        data['label'] = os.path.join(data_path, '{:06}-label.png'.format(i))
        database.append(data)

    return database


def _vote_centers(im_label, cls_indexes, center, poses, num_classes):
    """ 
        ref. PoseCNN
        compute the voting label image in 2D
        im_label refers to num_class 0 as background
    """

    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3*num_classes), dtype=np.float32)
    vertex_weights = np.zeros(vertex_targets.shape, dtype=np.float32)

    c = np.zeros((2, 1), dtype=np.float32)
    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        if len(x) > 0:
            ind = np.where(cls_indexes == i)[0] 
            c[0] = center[ind, 0]
            c[1] = center[ind, 1]
            z = poses[2, 3, ind]
            R = np.tile(c, (1, len(x))) - np.vstack((x, y))
            # compute the norm
            N = np.linalg.norm(R, axis=0) + 1e-10
            # normalization
            R = np.divide(R, np.tile(N, (2,1)))
            # assignment
            start = 3 * i
            end = start + 3
            vertex_targets[y, x, 3*i] = R[0,:]
            vertex_targets[y, x, 3*i+1] = R[1,:]
            vertex_targets[y, x, 3*i+2] = z
            vertex_weights[y, x, start:end] = 10.0

    return vertex_targets, vertex_weights



def get_a_sample(data_path=None, shuffle=True, batch_size=1, num_classes=1):

    b = 0
    index = -1
    dataset_indexes = prepare_dataset_indexes(data_path)

    # print len(dataset_indexes)

    if b == 0:
        # init arrays
        batch_rgb = np.zeros((batch_size, 480, 640, 3), dtype=np.float32)
        batch_center = np.zeros((batch_size, 480, 640, num_classes * 3), dtype=np.float32)            

    index = (index + 1) % len(dataset_indexes)
    if shuffle and index == 0:
        np.random.shuffle(dataset_indexes)

    data_rec = dataset_indexes[index]

    rgb_raw = pad_im(cv2.imread(data_rec["color"], cv2.IMREAD_UNCHANGED), 16)
    cv2.imwrite("rgb_raw.png",rgb_raw)

    rgb = rgb_raw.astype(np.float32, copy=True)
    mat = loadmat(data_rec["meta"])
    im_lbl = pad_im(cv2.imread(data_rec['label'], cv2.IMREAD_UNCHANGED), 16)

    batch_rgb[b,:,:,:] = rgb

    center_targets, center_weights = _vote_centers(im_lbl, mat['cls_indexes'], mat['center'], mat['poses'], num_classes)
    batch_center[b,:,:,:] = center_targets

    b = b+1

    if b >= batch_size:
        inputs = batch_rgb
        outputs = batch_center

        b = 0

        return (inputs, outputs)



def data_generator(data_path=None, shuffle=True, batch_size=1, num_classes=1):

    b = 0
    index = -1
    dataset_indexes = prepare_dataset_indexes(data_path)

    # print len(dataset_indexes)
    while True:

        if b == 0:
            # init arrays
            batch_rgb = np.zeros((batch_size, 480, 640, 3), dtype=np.float32)
            batch_center = np.zeros((batch_size, 480, 640, num_classes * 3), dtype=np.float32)       
            batch_center_weight = np.zeros((batch_size, 480, 640, 3 * num_classes), dtype=np.float32)

        index = (index + 1) % len(dataset_indexes)
        if shuffle and index == 0:
            np.random.shuffle(dataset_indexes)

        data_rec = dataset_indexes[index]

        rgb_raw = pad_im(cv2.imread(data_rec["color"], cv2.IMREAD_UNCHANGED), 16)
        rgb = rgb_raw.astype(np.float32, copy=True)
        mat = loadmat(data_rec["meta"])
        im_lbl = pad_im(cv2.imread(data_rec['label'], cv2.IMREAD_UNCHANGED), 16)

        batch_rgb[b,:,:,:] = rgb

        center_targets, center_weights = _vote_centers(im_lbl, mat['cls_indexes'], mat['center'], mat['poses'], num_classes)
        batch_center[b,:,:,:] = center_targets
        batch_center_weight[b,:,:,:] = center_weights

        b = b+1

        if b >= batch_size:
            inputs = [batch_rgb]
            # outputs = [batch_center]
            outputs = [np.concatenate([batch_center, batch_center_weight], -1)]

            b = 0

            yield (inputs, outputs)
        

    
############################################################
#  Main
############################################################

if __name__ == "__main__":


    md = vgg16convs_vertex_pred(shape=(480, 640, 3))


    # # execute only if run as a script
    # # main()

    # input = Input(shape=(480, 640, 3), name='the_name',dtype='float32')

    # mdw = vgg16convs_vertex_pred(input)
    # # print('model output shape {}'.format(mdw.the_model.output_shape))

    # # rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    # # sgd = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.9)
    # sgd = keras.optimizers.SGD(lr=0.1, momentum=0., decay=0.9)


    # mdw.the_model.compile(optimizer=sgd,
    #             loss=mdw.smooth_l1_loss_vertex,
    #             # loss='mean_squared_error',
    #             )#metrics=['accuracy'])



    # # # TEST FROM HERE
    # # # mdw.the_model.summary()              
    
    # # optim = mdw.the_model.optimizer
    # # ws = []
    # # grads = []
    # # grad_funs = []
    # # # print("mdw.the_model.input = {}".format(mdw.the_model.input))
    # # # print("mdw.the_model.output = {}".format(mdw.the_model.layers[-1].output))
    # # # print("targets[0] = {}".format(mdw.the_model.targets[0]))
    # # # input_tensors = [mdw.the_model.input[0],
    # # #                 #  mdw.the_model.sample_weights[0],
    # # #                  K.placeholder(shape=(None,None,None,9)),
    # # #                  K.learning_phase()]
    # # # print("the_model.sample_weights = {}".format(mdw.the_model.sample_weights[0]))
    # # # input_tensors = [K.placeholder(shape=(None, 480, 640, 3), dtype='float32'),
    # # #                 # K.placeholder(shape=(480, 640, 3), dtype='float32'),  # input shape
    # # #                 # K.placeholder(shape=(None,None,None,9)), # output shape
    # # #                 # K.placeholder(shape=(None)),

    # # #                 mdw.the_model.sample_weights[0],
    # # #                 # K.placeholder([3]),
                    
    # # #                 # K.placeholder(shape=(480,640,9)),
    # # #                 mdw.the_model.targets[0],
    # # #                 K.learning_phase()]

    # # # print(input_tensors)

    # # # get weights from train model
    # # # weights = mdw.the_model.trainable_weights
    # # # print("len(weights)", len(weights))
    # # # print("weights:")
    # # # for weight in weights:
    # # #     print(weight)
    # #     # print (K.is_keras_tensor(weight))
    # #     # print ("w name ={}".format(weight.name[:-2]))
    # # #     if mdw.the_model.get_layer(weight.name[:-2]).trainable:
    # # #         ws.append(weight)

    # # # weights = [weight for weight in weights if mdw.the_model.get_layer(weight.name[:-2]).trainable] 
    # # # get_gradients returns tensors
    # # # print("tt loss tensor {}".format(mdw.the_model.total_loss))
    # # # gradients = optim.get_gradients(mdw.the_model.total_loss, weights)

    # # # print (gradients)
    # # # print (gradients[0].op)
    # # # print (gradients[0].graph)
    # # # print ("gradients is keras tensor? {}".format(K.is_keras_tensor(gradients[0])))

    # # # get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    # # # print(get_gradients)

    # # # for w in mdw.the_model.trainable_weights:
    # # #     ws.append(w)
    # # #     grad = optim.get_gradients(mdw.the_model.total_loss, w)
    # # #     grads.append(grad)
    # # #     print(w)
    # # #     print(grad)
    # # #     get_gradients = K.function(inputs=input_tensors, outputs=grad)

    # num_classes = 3 # including the background as '0'

    # data_path = '/home/shawnle/Documents/Restore_PoseCNN/PoseCNN-master/data_syn_LOV/data_2_objs/'
    # # data_path = '/home/shawnle/Documents/Projects/PoseCNN-master/data/LOV/3d_train_data'
    # dat_gen = data_generator(data_path, num_classes=num_classes)

    # # ypred_shape = mdw.vertex_pred.get_shape()
    # ypred_shape = mdw.the_model.output_shape
    # print('ypred_shape: ', ypred_shape)
    # ytrue = tf.placeholder(tf.float32, shape=(None, ypred_shape[1], ypred_shape[2], ypred_shape[3]*2), name='ytrue')
    
    # # total_loss = mdw.smooth_l1_loss_vertex(ytrue, mdw.vertex_pred)
    # total_loss = mdw.smooth_l1_loss_vertex(ytrue, mdw.the_model.layers[-1].output)
    # optimizer = tf.train.MomentumOptimizer(0.001, 0.).minimize(total_loss)
    
    # with tf.Session() as sess:
    #     # tf.initializers.global_variables()
    #     tf.global_variables_initializer()
    #     for i in range(20):
    #         for step in range(10):
    #             inp, out = next(dat_gen)

    #             # print('length of inp: ', inp[0].shape, out[0].shape)
    #             if len(inp) == 0: 
    #                 print("empty data. Continue..")
    #                 continue

    #             feed_dict = { input : inp[0],
    #                           ytrue: out[0]
    #                         #  mdw.the_model.targets[0] : np.ones(shape=((1,480,640,9)), dtype=np.float32),
    #                         #  mdw.the_model.sample_weights[0] : np.ones((1), dtype=np.float32),
    #                         #  'dropout/keras_learning_phase:0' : 0                 
    #             }

    #             _, loss, inp_tensor = sess.run([optimizer, total_loss, input], 
    #                                         feed_dict= feed_dict)#{input: inp[0], mdw.the_model.output: out[0][...,:9], ytrue: out[0]})
    #             # print('let\'s assume ', loss, inp_tensor.shape)
    # # # mode = 'INFERENCE'
    # print('-------------------------------------')
    # print('it passes the sess run......')
    # mode = 'TRAIN'
    
    # # # get weights from pre-trained model
    # # test_model = load_model('my_model.h5')
    # # # weights = test_model.trainable_weights
    # # weights = test_model._collected_trainable_weights
    # # print("len(weights)", len(weights))

    # # print("tt loss tensor of test model {}".format(test_model.total_loss))
    # # test_gradients = test_model.optimizer.get_gradients(test_model.total_loss, weights)

    # # print("test_gradients =", test_gradients)


    # # # print("input = {}".format(test_model.inputs))
    # # # print("output = {}".format(test_model.outputs))

    # # a_sample = get_a_sample(data_path, num_classes=num_classes)
    # # print(np.shape(a_sample[0]))
    # # print(np.shape(a_sample[1]))
    # # # print("vertext_pred_target = {}".format(test_model.targets[0].name))
    # # # print("vertext_pred_target len = {}".format(len(test_model.targets)))
    # # # print("vertext_pred_target shape = {}".format(test_model.targets[0].shape))

    # # # g = tf.get_default_graph()
    # # # print("all ops = {}".format(g.get_operations()))


    # # sess = tf.Session()
    # # sess.run(tf.initializers.global_variables())

    # # # feed_dict = { input : np.ones(shape=((1,480,640,3)), dtype=np.float32),
    # # #              mdw.the_model.targets[0] : np.ones(shape=((1,480,640,9)), dtype=np.float32),
    # # #              mdw.the_model.sample_weights[0] : np.ones((1), dtype=np.float32),
    # # #              'dropout/keras_learning_phase:0' : 0                 
    # # # }
    # # # feed to train model
    # # # feed_dict = { input : a_sample[0],
    # # #              mdw.the_model.targets[0] : a_sample[1],
    # # #              mdw.the_model.sample_weights[0] : np.ones((1), dtype=np.float32),
    # # #              'dropout/keras_learning_phase:0' : 0                 
    # # # }
    # # # feed to pretrain model
    # # feed_dict = { test_model.input : a_sample[0],
    # #              test_model.targets[0] : a_sample[1],
    # #              test_model.sample_weights[0] : np.ones((1), dtype=np.float32),
    # #              'dropout/keras_learning_phase:0' : 0
    # # }

    # # # grad_val = sess.run([gradients], feed_dict=feed_dict)

    # # # print("ws={}".format(test_model.trainable_weights))
    # # # print("ws={}".format(test_model.trainable_weights[-8]))

    # # # print("l -9 =".format(test_model.layers[-9]))
    # # # print("l -9 weights=".format(test_model.layers[-9].get_weights()))

    # # print("\n\nws={}".format(test_model.trainable_weights))    
    # # print("\n\nws collected ={}".format(test_model._collected_trainable_weights))
    # # print("model total loss = {}".format(test_model.total_loss))
    # # print("\n\nlayers output ={}".format([l.output for l in test_model.layers]))

    # # class compute_optim_updates():
    # #     def __init__(self, test_model, grads, lr_in, momentum, decay_in, init_decay_in=0.):

    # #         self.iterations = K.variable(0, dtype='int64', name='iterations')
    # #         self.nesterov = False
    # #         self.lr = K.variable(lr_in, name='lr')
    # #         self.decay = decay_in
    # #         self.initial_decay = init_decay_in
    # #         self.momentum = K.variable(momentum, name='momentum')

    # #     def get_updates(self):

    # #         self.updates = [state_ops.assign_add(self.iterations, 1)]
    # #         # self.updates = []
 
    # #         lr = self.lr
    # #         if self.initial_decay > 0:
    # #             lr = lr * (  # pylint: disable=g-no-augmented-assignment
    # #                 1. / (1. + self.decay * math_ops.cast(self.iterations,
    # #                                                         K.dtype(self.decay))))

    # #         params = test_model._collected_trainable_weights
    # #         shapes = [K.int_shape(p) for p in params]
    # #         moments = [K.zeros(shape) for shape in shapes]
    # #         self.weights = [self.iterations] + moments
    # #         for p, g, m in zip(params, grads, moments):
    # #             v = self.momentum * m - lr * g  # velocity
    # #             self.updates.append(state_ops.assign(m, v))
    # #             # self.updates.append(v)

    # #             if self.nesterov:
    # #                 new_p = p + self.momentum * v - lr * g
    # #             else:
    # #                 new_p = p + v

    # #             # Apply constraints.
    # #             if getattr(p, 'constraint', None) is not None:
    # #                 new_p = p.constraint(new_p)

    # #             self.updates.append(state_ops.assign(p, new_p))
    # #             # self.updates.append(new_p)

    # #         return self.updates

    # # cou = compute_optim_updates(test_model, test_gradients, test_model.optimizer.lr,
    # #                                                         test_model.optimizer.momentum,
    # #                                                         test_model.optimizer.decay)

    # # # print("compute optim updates = ",cou.get_updates())  

    # # grad_val, model_outputs, updates_val = sess.run([test_gradients, test_model.outputs, cou.get_updates()], feed_dict=feed_dict)

    # # print("len(model_outputs) =", len(model_outputs))
    # # print("shape(model_outputs[0])=", model_outputs[0].shape)
    # # print("shape(target) = ", a_sample[1].shape)

    # # delta_L = a_sample[1] - model_outputs[0]
    # # print("delta_L shape = ", delta_L.shape)
    # # print("cls 1 max loss vx= ", np.amax(delta_L[0,:,:,3]))
    # # print("cls 1 max loss vy= ", np.amax(delta_L[0,:,:,4]))
    # # print("\n\nws[-2]={}".format(test_model.trainable_weights[-2]))

    # # exit()
    
    # # # get the trained weight values by sess.run
    # # W_L = sess.run ([test_model.trainable_weights[-2]])
    # # print("W_L = ", W_L)

    # # from im2col import im2col_indices
    # # delta_L_T = np.transpose(delta_L, (3,0,1,2))
    # # print ("delta_L_T shape = ", delta_L_T.shape)
    # # delta_L_T_2col = im2col_indices(delta_L_T, 1, 1, padding=0, stride=1)
    # # print("delta_L_2col shape = ", delta_L_T_2col.shape)

    # # # note 0: background class
    # # cls_id = 1
    # # np.save("mdl_out_vx.npy", np.squeeze(model_outputs[0])[:,:, 3*cls_id+0])   
    # # np.save("mdl_out_vy.npy", np.squeeze(model_outputs[0])[:,:, 3*cls_id+1])

    # # writer = tf.summary.FileWriter('.')
    # # writer.add_graph(tf.get_default_graph())
    # # writer.flush()

    # # exit()

    # # for grad in grad_val:
    # #     for grad_ in grad:
    # #         grad_ = np.squeeze(grad_)
    # #         print('grad shape =', grad_.shape)

    # #         if len(grad_.shape) > 1:
    # #             for i in xrange(grad_.shape[0]):
    # #                 for j in xrange(grad_.shape[1]):
    # #                     if (i > j or i < j) and abs(grad_[i,j]) > 0.00000001 :
    # #                         print((i,j))                        
    # #         else :
    # #             print("bias {}", grad_)
    # #             for i in xrange(grad_.shape[0]):
    # #                 if grad_[i] < 0:
    # #                     print ("grad_[{}]={}".format(i,grad_[i]))
            
    # #         if grad_.shape[0] > grad_.shape[1]:
    # #             min_dim = grad_.shape[1]
    # #         else:
    # #             min_dim = grad_.shape[0]

    # #         for i in xrange(min_dim):
    # #             print("grad_[{},{}]={}".format(i,i,grad_[i,i]))

    # #         print('grad norm =', LA.norm(grad_))

    # # exit()

    # # # rgb = np.squeeze(a_sample[0], axis=(1,))
    # # # print(np.shape(rgb))
    # # inputs=[#np.squeeze(a_sample[0]),
    # #         np.ones(shape=((1,480,640,3)), dtype=np.float32), 
    # #         np.ones((1), dtype=np.float32),
    # #         np.ones(shape=((1,480,640,9)), dtype=np.float32),
    # #         # a_sample[1],
    # #         0]
    # # # print(zip(test_model.trainable_weights, get_gradients(inputs)))
    # # print(get_gradients(inputs))




    # # print(mdw.the_model.sample_weights[0])
    # # print(np.shape(mdw.the_model.sample_weights[0]))
    # # exit()

    # # # print(test_model.get_layer('score_conv4_vertex').get_weights())
    # # # print(np.shape(test_model.get_layer('score_conv4_vertex').get_weights()[0]))
    # # # print(len(ws))
    # # # print(len(grads))

    # # for layer in test_model.layers:
    # #     print (layer.name)
    # #     print (layer.input)
    # #     print (layer.output)

    # # tf_sess = K.get_session()
    # # graph = K.get_default_graph()
    # # print(graph.get_operations())
    # # # tf_sess.run([gradients])
    # # # tf_sess.run([grads[2]], feed_dict={'score_conv4_vertex/kernel:0': test_model.get_layer('score_conv4_vertex').get_weights()[0],
    # # #                                     'vertex_pred_sample_weights' : np.ones(np.shape(test_model.get_layer('score_conv4_vertex').get_weights()[0])) })
    # # # print(  grads[2](  [test_model.get_layer('score_conv4_vertex').get_weights()[0]]  )  )

    # # exit()

    # # # TEST TO HERE





    # if mode == 'TRAIN' :
    #     epoch_num = 30
    #     batch_size = 3
    #     # num_samples = 2822
    #     num_samples = 12
    #     steps_per_epoch = int(ceil(num_samples / batch_size))
    #     mdw.the_model.fit_generator(dat_gen, steps_per_epoch=steps_per_epoch, epochs=epoch_num, verbose=2)
    #     mdw.the_model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

    # if mode == 'INFERENCE':

    #     test_model = load_model('my_model.h5')

    #     print(test_model.get_weights())
    #     print(type(test_model.get_weights()))
    #     print(np.shape(test_model.get_weights()))
    #     print(np.shape(test_model.get_weights()[0]))
    #     print(test_model.get_weights()[0])
    #     print(np.shape(test_model.get_weights()[0]))
    #     print(type(test_model.get_weights()[0]))

    #     for lr in test_model.layers:
    #         print(lr.name)
        
    #     print(test_model.get_layer('score_conv4_vertex').get_weights())

    #     num_test = 1
    #     test_batch = np.zeros((num_test, 480, 640, 3), dtype=np.float32)

    #     sample_ = pad_im(cv2.imread(os.path.join(data_path, '{:06}-color.png'.format(150)), cv2.IMREAD_UNCHANGED), 16)
    #     sample = sample_.astype(np.float32, copy=True)

    #     test_batch[0,:,:,:] = sample
    #     pred = test_model.predict(test_batch)

    #     print (pred.shape)
    #     cv2.imwrite('vx0.png', pred[0,:,:,3])
    #     cv2.imwrite('vy0.png', pred[0,:,:,4])
    #     cv2.imwrite('Z0.png', pred[0,:,:,5])
    #     cv2.imwrite('vx1.png', pred[0,:,:,6])
    #     cv2.imwrite('vy1.png', pred[0,:,:,7])
    #     cv2.imwrite('Z1.png', pred[0,:,:,8])
