from __future__ import print_function

import _init_paths

import glob
import os

from math import ceil
import numpy as np
from scipy.io import loadmat

# from gt_data_layer.minibatch import _process_label_image
import hard_label_layer.hard_label_op as hard_label_op
import hard_label_layer.hard_label_op_grad

import cv2

import tensorflow as tf
from tensorflow.python.ops import state_ops

from fcn.config import cfg

from numpy import linalg as LA

from utils.blob import chromatic_transform, unpad_im

import matplotlib.pyplot as plt


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

def hard_label(input, threshold, name):
    return hard_label_op.hard_label(input[0], input[1], threshold, name=name)

def add(inputs, name):
    if isinstance(inputs[0], tuple):
        inputs[0] = inputs[0][0]

    if isinstance(inputs[1], tuple):
        inputs[1] = inputs[1][0]

    return tf.add_n(inputs, name=name)

def dropout(input, keep_prob, name):
    if isinstance(input, tuple):
        input = input[0]
    return tf.nn.dropout(input, keep_prob, name=name)

def argmax_2d(input, name):
    return tf.to_int32(tf.argmax(input, 3, name))

def softmax_high_dimension(input, num_classes, name):

    # only use the first input
    if isinstance(input, tuple):
        input = input[0]
    input_shape = input.get_shape()
    ndims = input_shape.ndims
    array = np.ones(ndims)
    array[-1] = num_classes

    m = tf.reduce_max(input, reduction_indices=[ndims-1], keep_dims=True)
    multiples = tf.convert_to_tensor(array, dtype=tf.int32)
    e = tf.exp(tf.subtract(input, tf.tile(m, multiples)))
    s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
    return tf.div(e, tf.tile(s, multiples))

def log_softmax_high_dimension(input, num_classes, name):
    # only use the first input
    if isinstance(input, tuple):
        input = input[0]
    input_shape = input.get_shape()
    ndims = input_shape.ndims
    array = np.ones(ndims)
    array[-1] = num_classes

    m = tf.reduce_max(input, reduction_indices=[ndims-1], keep_dims=True)
    multiples = tf.convert_to_tensor(array, dtype=tf.int32)
    d = tf.subtract(input, tf.tile(m, multiples))
    e = tf.exp(d)
    s = tf.reduce_sum(e, reduction_indices=[ndims-1], keep_dims=True)
    return tf.subtract(d, tf.log(tf.tile(s, multiples)))
    
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
    return tf.to_float(output)

def make_deconv_filter(name, f_shape, trainable=True):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=trainable)
    return var

def deconv(input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, padding=DEFAULT_PADDING, trainable=True):
    validate_padding(padding)
    c_i = input.get_shape()[-1]
    # print("input shape =", input.get_shape())
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Compute shape out of input
        # in_shape = tf.shape(input)        
        in_shape = input.get_shape()
        h = in_shape[1] * s_h
        w = in_shape[2] * s_w
        # new_shape = [in_shape[0], h, w, c_o]  # cannot use. Why?
        new_shape = [tf.shape(input)[0], h, w, c_o]
        output_shape = tf.stack(new_shape)

        # filter
        f_shape = [k_h, k_w, c_o, c_i]
        weights = make_deconv_filter('weights', f_shape, trainable)
    return tf.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding, name=scope.name)

def load(data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        # data_dict = np.load(data_path).item()
        # link: https://stackoverflow.com/questions/38316283/trouble-using-numpy-load
        # data_dict = np.load(data_path, allow_pickle=True, encoding='latin1').item()
        data_dict = np.load(data_path).item()
        for op_name in data_dict:
            print(op_name) 
            with tf.variable_scope(op_name, reuse=True):
                # for param_name, data in data_dict[op_name].iteritems():
                # link: https://stackoverflow.com/questions/30418481/error-dict-object-has-no-attribute-iteritems
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        print (op_name + ' ' + param_name + ' assigned')
                    except ValueError:
                        if not ignore_missing:
                            raise
            # try to assign dual weights
            with tf.variable_scope(op_name+'_p', reuse=True):
                #for param_name, data in data_dict[op_name].iteritems():
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        print (op_name + '_p ' + param_name + ' assigned')
                    except ValueError:
                        if not ignore_missing:
                            raise

            with tf.variable_scope(op_name+'_d', reuse=True):
                # for param_name, data in data_dict[op_name].iteritems():
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

def _process_label_image(label_image, class_colors, class_weights):
    """
    change label image to label index
    """
    height = label_image.shape[0]
    width = label_image.shape[1]
    num_classes = len(class_colors)
    label_index = np.zeros((height, width, num_classes), dtype=np.float32)

    if len(label_image.shape) == 3:
        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I[0], I[1], i] = class_weights[i]
    else:
        for i in xrange(len(class_colors)):
            I = np.where(label_image == i)
            label_index[I[0], I[1], i] = class_weights[i]
    
    return label_index


def labels_to_image_tensor(labels):
    '''
      adapt from PCNN
    '''
    class_colors = tf.constant([(0,0,0), (255, 255, 0), (255, 0, 255)])

    height = labels.get_shape().as_list()[1]
    width = labels.get_shape().as_list()[2]
    image_r = tf.zeros((height, width), dtype=tf.float32)
    image_g = tf.zeros((height, width), dtype=tf.float32)
    image_b = tf.zeros((height, width), dtype=tf.float32)

    # for i in xrange(1,len(class_colors)):
    for i in xrange(1,3):
        color = class_colors[i]
        I = tf.where(tf.equal(labels,i)) #labels == i
        print("I ={}, i={}".format(I,i))
        # image_r[I] = color[0]
        # image_g[I] = color[1]
        # image_b[I] = color[2]

    # image = np.stack((image_r, image_g, image_b), axis=-1)

    return tf.scalar_mul(tf.constant([50], dtype=tf.int32, shape=[]), labels) # image.astype(np.uint8)


############################################################
#  Network Class
############################################################

class vgg16convs_vertex_pred():

    # def __init__(self, input):
    def __init__(self, shape=(None,None,None), trainable=True):

        # self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.input = tf.placeholder(tf.float32, shape=[None, shape[0], shape[1], shape[2]])
        self.gt_label_2d = tf.placeholder(tf.int32, shape=[None, None, None])
        self.trainable = trainable
        self.num_units = 64
        self.keep_prob_queue = 0.5 #1. 
        self.threshold_label = .5
        self.scale = 1.
        self.vertex_reg = 1
        self.num_classes = 3

        self.layers = []
        self.layer_dict = dict()
        self.layers.append(self.input)

        # print("layers = ", self.layers)

        self.build()

        self.tensorboard_rep()


    def make_tensor_img(self, inp):
        '''
            0-255 normalization: (x-min)*255 / (max-min)
        '''
        # return tf.div((inp - tf.reduce_min(inp))*tf.constant(255., dtype=tf.float32), (tf.reduce_max(inp) - tf.reduce_min(inp)) )
        #return tf.cast(tf.div((inp - tf.reduce_min(inp))*tf.constant(255., dtype=tf.float32), (tf.reduce_max(inp) - tf.reduce_min(inp)) ), tf.uint8)
        return tf.div((inp - tf.reduce_min(inp))*tf.constant(255., dtype=tf.float32), (tf.reduce_max(inp) - tf.reduce_min(inp)))


    def tensorboard_rep(self):

        print("[tensorboard_rep] running...")
        rep_list = ['vertex_pred', 'add_score_vertex', 'score_conv4_vertex', 'score_conv5_vertex', 'conv5_3', 'conv4_3']

        with tf.name_scope('summaries') as self.scope:
        #     # for rep in rep_list:
            shape = self.layer_dict['vertex_pred'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['vertex_pred'], [0,0,0,3], [1, shape[1], shape[2], 1]))
            tf.summary.image('vertex_pred', tensor_img)
            # tf.summary.scalar('vertex_pred_max', tf.math.reduce_max(self.layer_dict['vertex_pred']))
            # tf.summary.scalar('vertex_pred_min', tf.math.reduce_min(self.layer_dict['vertex_pred']))

            upscore = self.layer_dict['upscore'] 
            tf.summary.scalar('upscore_sum_abs', tf.reduce_sum(tf.abs(upscore)))
            score = self.layer_dict['score'] 
            tf.summary.scalar('score_sum_abs', tf.reduce_sum(tf.abs(score)))
            prob = self.layer_dict['prob'] 
            tf.summary.scalar('prob_sum_abs', tf.reduce_sum(tf.abs(prob)))
            prob_normalized = self.layer_dict['prob_normalized'] 
            tf.summary.scalar('prob_normalized_sum_abs', tf.reduce_sum(tf.abs(prob_normalized)))
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['prob_normalized'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('prob_normalized_0', tensor_img)
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['prob_normalized'], [0,0,0,1], [1, shape[1], shape[2], 1]))
            tf.summary.image('prob_normalized_1', tensor_img)
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['prob_normalized'], [0,0,0,2], [1, shape[1], shape[2], 1]))
            tf.summary.image('prob_normalized_2', tensor_img)

            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['vertex_pred'], [0,0,0,4], [1, shape[1], shape[2], 1]))
            tf.summary.image('vertex_pred', tensor_img)
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['vertex_pred'], [0,0,0,6], [1, shape[1], shape[2], 1]))
            tf.summary.image('vertex_pred', tensor_img)
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['vertex_pred'], [0,0,0,7], [1, shape[1], shape[2], 1]))
            tf.summary.image('vertex_pred', tensor_img)
                    
            shape = self.layer_dict['add_score_vertex'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['add_score_vertex'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('add_score_vertex', tensor_img)

            shape = self.layer_dict['score_conv4_vertex'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['score_conv4_vertex'], [0, 0, 0, 0], [1, shape[1], shape[2], 1]))
            tf.summary.image('score_conv4_vertex', tensor_img)

            shape = self.layer_dict['score_conv5_vertex'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['score_conv5_vertex'], [0, 0, 0, 0], [1, shape[1], shape[2], 1]))
            tf.summary.image('score_conv5_vertex', tensor_img)

            shape = self.layer_dict['conv5_3'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv5_3'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv5_3', tensor_img)

            shape = self.layer_dict['conv4_3'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv4_3'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv4_3', tensor_img)

            shape = self.layer_dict['conv1_1'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv1_1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv1_1', tensor_img)

            shape = self.layer_dict['conv1_2'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv1_2'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv1_2', tensor_img)

            shape = self.layer_dict['pool1'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['pool1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('pool1', tensor_img)

            shape = self.layer_dict['conv2_1'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv2_1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv2_1', tensor_img)

            shape = self.layer_dict['conv2_2'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv2_2'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv2_2', tensor_img)

            shape = self.layer_dict['pool2'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['pool2'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('pool2', tensor_img)

            shape = self.layer_dict['conv3_1'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv3_1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv3_1', tensor_img)

            shape = self.layer_dict['conv3_2'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv3_2'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv3_2', tensor_img)

            shape = self.layer_dict['conv3_3'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv3_3'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv3_3', tensor_img)            
            with tf.variable_scope("conv3_3", reuse=True) as scope:
                W = tf.get_variable("weights")
                b = tf.get_variable("biases")
                tf.summary.scalar('conv3_1_W_sum', tf.reduce_sum(tf.abs(W)))
                tf.summary.scalar('conv3_1_b_sum', tf.reduce_sum(tf.abs(b)))

            shape = self.layer_dict['pool3'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['pool3'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('pool3', tensor_img)    

            shape = self.layer_dict['conv4_1'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv4_1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv4_1', tensor_img)
            tf.summary.scalar('conv4_1_sum', tf.reduce_sum(self.layer_dict['conv4_1']))
            tf.summary.scalar('conv4_1_image_max', tf.reduce_max(tensor_img))
            tf.summary.scalar('conv4_1_image_min', tf.reduce_min(tensor_img))
            with tf.variable_scope("conv4_1", reuse=True) as scope:
                W = tf.get_variable("weights")
                b = tf.get_variable("biases")
                tf.summary.scalar('conv4_1_W_sum', tf.reduce_sum(tf.abs(W)))
                tf.summary.scalar('conv4_1_b_sum', tf.reduce_sum(tf.abs(b)))

            # conv4_1_W = tf.get_variable("my_variable", [1, 2, 3])
            # print('all tensors = \n',[n.name for n in tf.get_default_graph().as_graph_def().node])

            shape = self.layer_dict['conv4_2'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv4_2'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv4_2', tensor_img)

            shape = self.layer_dict['pool4'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['pool4'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('pool4', tensor_img)
            tf.summary.scalar('pool4_sum_abs', tf.reduce_sum(tf.abs(self.layer_dict['pool4'])))

            shape = self.layer_dict['conv5_1'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv5_1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv5_1', tensor_img)
            tf.summary.scalar('conv5_1_sum', tf.reduce_sum(self.layer_dict['conv5_1']))
            with tf.variable_scope("conv5_1", reuse=True) as scope:
                W = tf.get_variable("weights")
                b = tf.get_variable("biases")
                tf.summary.scalar('conv5_1_W_sum', tf.reduce_sum(tf.abs(W)))
                tf.summary.scalar('conv5_1_b_sum', tf.reduce_sum(tf.abs(b)))

            shape = self.layer_dict['conv5_2'].shape
            tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['conv5_2'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            tf.summary.image('conv5_2', tensor_img)

            # shape = self.layer_dict['upscore_vertex_conv'].shape
            # tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['upscore_vertex_conv'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            # tf.summary.image('upscore_vertex_conv', tensor_img)

            # shape = self.layer_dict['upscore_vertex_conv1'].shape
            # tensor_img = self.make_tensor_img(tf.slice(self.layer_dict['upscore_vertex_conv1'], [0,0,0,0], [1, shape[1], shape[2], 1]))
            # tf.summary.image('upscore_vertex_conv1', tensor_img)


    def backend_debug_print(self, x):
        K.print_tensor(x, message='hello_print') # , [tf.shape(x)]
        return x

    def loss_cross_entropy_single_frame(self, scores, labels):
        """
        scores: a tensor [batch_size, height, width, num_classes]
        labels: a tensor [batch_size, height, width, num_classes]
        """

        with tf.name_scope('loss'):
            cross_entropy = -tf.reduce_sum(labels * scores, reduction_indices=[3])
            loss = tf.div(tf.reduce_sum(cross_entropy), tf.reduce_sum(labels)+1e-10)

        return loss

    def smooth_l1_loss_vertex(self, vertex_pred, vertex_targets, vertex_weights, sigma=1.0):
    # def smooth_l1_loss_vertex(self, vertex_targets_, vertex_pred):

        # vertex_targets, vertex_weights = vertex_targets_[...,:9],  vertex_targets_[...,9:] 

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
        self.layers.append([conv1_1, conv1_2, pool1])
        self.layer_dict['conv1_1'] = conv1_1
        self.layer_dict['conv1_2'] = conv1_2
        self.layer_dict['pool1'] = pool1

        conv2_1 = conv(pool1, 3, 3, 128, 1, 1, name='conv2_1', c_i=64, trainable=self.trainable)
        conv2_2 = conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2', c_i=128, trainable=self.trainable)
        pool2 = max_pool(conv2_2, 2, 2, 2, 2, name='pool2')
        self.layers.append([conv2_1, conv2_2, pool2])
        self.layer_dict['conv2_1'] = conv2_1
        self.layer_dict['conv2_2'] = conv2_2
        self.layer_dict['pool2'] = pool2

        conv3_1 = conv(pool2, 3, 3, 256, 1, 1, name='conv3_1', c_i=128, trainable=self.trainable)
        conv3_2 = conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2', c_i=256, trainable=self.trainable)
        conv3_3 = conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3', c_i=256, trainable=self.trainable)
        pool3 = max_pool(conv3_3, 2, 2, 2, 2, name='pool3')
        self.layers.append([conv3_1, conv3_2, conv3_3, pool3])
        self.layer_dict['conv3_1'] = conv3_1
        self.layer_dict['conv3_2'] = conv3_2
        self.layer_dict['conv3_3'] = conv3_3
        self.layer_dict['pool3'] = pool3

        conv4_1 = conv(pool3, 3, 3, 512, 1, 1, name='conv4_1', c_i=256, trainable=self.trainable)
        conv4_2 = conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2', c_i=512, trainable=self.trainable)
        conv4_3 = conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3', c_i=512, trainable=self.trainable)
        pool4 = max_pool(conv4_3, 2, 2, 2, 2, name='pool4')
        self.layers.append([conv4_1, conv4_2, conv4_3, pool4])
        self.layer_dict['conv4_1'] = conv4_1
        self.layer_dict['conv4_2'] = conv4_2
        self.layer_dict['conv4_3'] = conv4_3
        self.layer_dict['pool4'] = pool4

        conv5_1 = conv(pool4, 3, 3, 512, 1, 1, name='conv5_1', c_i=512, trainable=self.trainable)
        conv5_2 = conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2', c_i=512, trainable=self.trainable)
        conv5_3 = conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3', c_i=512, trainable=self.trainable)
        self.layers.append([conv5_1, conv5_2, conv5_3])
        self.layer_dict['conv5_1'] = conv5_1
        self.layer_dict['conv5_2'] = conv5_2
        self.layer_dict['conv5_3'] = conv5_3

        score_conv5 = conv(conv5_3, 1, 1, self.num_units, 1, 1, name='score_conv5', c_i=512)
        upscore_conv5 = deconv(score_conv5, 4, 4, self.num_units, 2, 2, name='upscore_conv5', trainable=False)
        self.layers.append([score_conv5, upscore_conv5])
        self.layer_dict['score_conv5'] = score_conv5
        self.layer_dict['upscore_conv5'] = upscore_conv5

        score_conv4 = conv(conv4_3, 1, 1, self.num_units, 1, 1, name='score_conv4', c_i=512)
        self.layer_dict['score_conv4'] = score_conv4

        add_score = add([score_conv4, upscore_conv5], name='add_score')
        dropout_ = dropout(add_score, self.keep_prob_queue, name='dropout')
        upscore = deconv(dropout_, int(16*self.scale), int(16*self.scale), self.num_units, int(8*self.scale), int(8*self.scale), name='upscore', trainable=False)
        self.layers.append([score_conv4, add_score, dropout_, upscore])
        self.layer_dict['add_score'] = add_score
        self.layer_dict['dropout'] = dropout_
        self.layer_dict['upscore'] = upscore

        score = conv(upscore, 1, 1, self.num_classes, 1, 1, name='score', c_i=self.num_units)
        prob = log_softmax_high_dimension(score, self.num_classes, name='prob')
        self.layer_dict['prob'] = prob
        self.layer_dict['score'] = score

        prob_normalized = softmax_high_dimension(score, self.num_classes, name='prob_normalized')
        label_2d = argmax_2d(prob_normalized, name='label_2d')
        self.layer_dict['label_2d'] = label_2d
        self.layer_dict['prob_normalized'] = prob_normalized
        
        gt_label_weight = hard_label([prob_normalized, self.gt_label_2d], threshold=self.threshold_label, name='gt_label_weight')
        self.layer_dict['gt_label_weight'] = gt_label_weight

        if self.vertex_reg : 
            score_conv5_vertex = conv(conv5_3, 1, 1, 128, 1, 1, name='score_conv5_vertex', relu=False, c_i=512)
            upscore_conv5_vertex = deconv(score_conv5_vertex, 4, 4, 128, 2, 2, name='upscore_conv5_vertex', trainable=True)
            self.layers.append([score_conv5_vertex, upscore_conv5_vertex])
            self.layer_dict['score_conv5_vertex'] = score_conv5_vertex

            score_conv4_vertex = conv(conv4_3, 1, 1, 128, 1, 1, name='score_conv4_vertex', relu=False, c_i=512)
            self.layers.append(score_conv4_vertex)
            self.layer_dict['score_conv4_vertex'] = score_conv4_vertex

            add_score_vertex = add([score_conv4_vertex, upscore_conv5_vertex], name='add_score_vertex')
            dropout_vertex = dropout(add_score_vertex, keep_prob=self.keep_prob_queue, name='dropout_vertex')

            upscore_vertex = deconv(dropout_vertex, int(16*self.scale), int(16*self.scale), 128, int(8*self.scale), int(8*self.scale), name='upscore_vertex', trainable=True)
            # upscore_vertex = deconv(dropout_vertex, int(16*self.scale), int(16*self.scale), 128, int(8*self.scale), int(8*self.scale), name='upscore_vertex', trainable=False)
            # upscore_vertex_conv = conv(upscore_vertex, 1, 1, 128, 1, 1, name='upscore_vertex_conv', relu=False, c_i=128)
            # upscore_vertex1 = deconv(upscore_vertex_conv, int(4*self.scale), int(4*self.scale), 128, int(2*self.scale), int(2*self.scale), name='upscore_vertex1', trainable=False)
            # upscore_vertex_conv1 = conv(upscore_vertex1, 1, 1, 128, 1, 1, name='upscore_vertex_conv1', relu=False, c_i=128)
            # upscore_vertex2 = deconv(upscore_vertex_conv1, int(4*self.scale), int(4*self.scale), 128, int(2*self.scale), int(2*self.scale), name='upscore_vertex2', trainable=False)
            # upscore_vertex_conv2 = conv(upscore_vertex2, 1, 1, 128, 1, 1, name='upscore_vertex_conv2', relu=False, c_i=128)

            self.layers.append([add_score_vertex, dropout_vertex, upscore_vertex])
            self.layer_dict['add_score_vertex'] = add_score_vertex            
            self.layer_dict['dropout_vertex'] = dropout_vertex    
            self.layer_dict['upscore_vertex'] = upscore_vertex
            # self.layer_dict['upscore_vertex1'] = upscore_vertex1
            # self.layer_dict['upscore_vertex2'] = upscore_vertex2
            # self.layer_dict['upscore_vertex_conv'] = upscore_vertex_conv
            # self.layer_dict['upscore_vertex_conv1'] = upscore_vertex_conv1

            vertex_pred = conv(upscore_vertex, 1, 1, 3 * self.num_classes, 1, 1, name='vertex_pred', relu=False, c_i=128)
            # vertex_pred = conv(upscore_vertex2, 1, 1, 3 * self.num_classes, 1, 1, name='vertex_pred', relu=False, c_i=128)
            self.output = vertex_pred
            self.layers.append(self.output)
            self.layer_dict['vertex_pred'] = vertex_pred

        for l, i in enumerate(self.layers, 0):
            print("layers " + str(l) + " " + str(i))
        print("model input = ", self.input)
        print("model output = ", self.output)
        print("layers dict: ", self.layer_dict)


def restore(session, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                if var.name.split(':')[0] in saved_shapes])

        var_name_to_var = {var.name : var for var in tf.global_variables()}
        restore_vars = []
        restored_var_names = set()
        print('Restoring:')
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for var_name, saved_var_name in var_names:
                if 'global_step' in var_name:
                    continue
                if 'Variable' in var_name:
                    continue
                curr_var = var_name_to_var[var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                    print(str(saved_var_name))
                    restored_var_names.add(saved_var_name)
                else:
                    print('Shape mismatch for var', saved_var_name, 'expected', var_shape, 'got', saved_shapes[saved_var_name])
        ignored_var_names = sorted(list(set(saved_shapes.keys()) - restored_var_names))
        if len(ignored_var_names) == 0:
            print('Restored all variables')
        else:
            print('Did not restore:' + '\n\t'.join(ignored_var_names))

        if len(restore_vars) > 0:
            saver = tf.train.Saver(restore_vars)
            saver.restore(session, save_file)
        print('Restored %s' % save_file)

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
    for i in range(1, num_classes):
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
    '''
        num_classes includes the background class
    '''

    # print('under development. Refer to data_generator for a reference')
    # exit()

    b = 0
    index = -1
    dataset_indexes = prepare_dataset_indexes(data_path)
    class_colors = [(0,0,0), (255, 255, 0), (255, 0, 255)]  
    class_weights = [1, 1, 1]

    assert len(class_colors) == num_classes and len(class_weights) == num_classes, "num_classes = " + str(num_classes) 
    print ('dataset size = ' + str(len(dataset_indexes)))

    if b == 0:
        # init arrays
        batch_rgb = np.zeros((batch_size, 480, 640, 3), dtype=np.float32)
        batch_lbl = np.zeros((batch_size, 480, 640, num_classes), dtype=np.float32)
        batch_center = np.zeros((batch_size, 480, 640, num_classes * 3), dtype=np.float32)       
        batch_center_weight = np.zeros((batch_size, 480, 640, 3 * num_classes), dtype=np.float32)

    index = (index + 1) % len(dataset_indexes)
    if shuffle and index == 0:
        np.random.shuffle(dataset_indexes)

    data_rec = dataset_indexes[index]

    read_im = cv2.imread(data_rec["color"], cv2.IMREAD_UNCHANGED)
    assert read_im is not None
    rgb_raw = pad_im(read_im, 16)
    rgb = rgb_raw

    if cfg.TRAIN.CHROMATIC:
        rgb = chromatic_transform(rgb)

    rgb = rgb_raw.astype(np.float32, copy=True)
    rgb -= cfg.PIXEL_MEANS
    mat = loadmat(data_rec["meta"])
    im_lbl = pad_im(cv2.imread(data_rec['label'], cv2.IMREAD_UNCHANGED), 16)

    im_cls = _process_label_image(im_lbl, class_colors, class_weights)

    batch_rgb[b,:,:,:] = rgb
    batch_lbl[b,...] = im_cls #im_lbl

    center_targets, center_weights = _vote_centers(im_lbl, mat['cls_indexes'], mat['center'], mat['poses'], num_classes)
    batch_center[b,:,:,:] = center_targets
    batch_center_weight[b,:,:,:] = center_weights

    b = b+1

    if b >= batch_size:
        inputs = [batch_rgb, batch_lbl]
        outputs = [batch_center, batch_center_weight]

        b = 0

        return (inputs, outputs)



def data_generator(data_path=None, shuffle=True, batch_size=1, num_classes=1):
    '''
        num_classes includes the background class
    '''

    b = 0
    index = -1
    dataset_indexes = prepare_dataset_indexes(data_path)
    class_colors = [(0,0,0), (255, 255, 0), (255, 0, 255)]  
    class_weights = [1, 1, 1]

    assert len(class_colors) == num_classes and len(class_weights) == num_classes, "num_classes = " + str(num_classes) 
    print ('dataset size = ' + str(len(dataset_indexes)))
    while True:

        if b == 0:
            # init arrays
            batch_rgb = np.zeros((batch_size, 480, 640, 3), dtype=np.float32)
            batch_lbl = np.zeros((batch_size, 480, 640, num_classes), dtype=np.float32)
            batch_center = np.zeros((batch_size, 480, 640, num_classes * 3), dtype=np.float32)       
            batch_center_weight = np.zeros((batch_size, 480, 640, 3 * num_classes), dtype=np.float32)

        index = (index + 1) % len(dataset_indexes)
        if shuffle and index == 0:
            np.random.shuffle(dataset_indexes)

        data_rec = dataset_indexes[index]

        read_im = cv2.imread(data_rec["color"], cv2.IMREAD_UNCHANGED)
        if read_im is None:
            continue
        rgb_raw = pad_im(read_im, 16)

        if cfg.TRAIN.CHROMATIC:
            rgb_raw = chromatic_transform(rgb_raw)

        rgb = rgb_raw.astype(np.float32, copy=True)
        rgb -= cfg.PIXEL_MEANS
        mat = loadmat(data_rec["meta"])
        im_lbl = pad_im(cv2.imread(data_rec['label'], cv2.IMREAD_UNCHANGED), 16)

        im_cls = _process_label_image(im_lbl, class_colors, class_weights)

        batch_rgb[b,:,:,:] = rgb
        batch_lbl[b,...] = im_cls

        center_targets, center_weights = _vote_centers(im_lbl, mat['cls_indexes'], mat['center'], mat['poses'], num_classes)
        batch_center[b,:,:,:] = center_targets
        batch_center_weight[b,:,:,:] = center_weights

        b = b+1

        if b >= batch_size:
            inputs = [batch_rgb, batch_lbl]
            outputs = [batch_center, batch_center_weight]
            # outputs = [np.concatenate([batch_center, batch_center_weight], -1)]

            b = 0

            yield (inputs, outputs)
        

    
############################################################
#  Main
############################################################

if __name__ == "__main__":

    mode = 'train'    # 'test' # 

    rgb_shape = (480, 640, 3)
    md = vgg16convs_vertex_pred(shape=rgb_shape, trainable=True)   # trainable=False

    num_classes = 3 # including the background as '0'

    batch_size = 1

    # data_path = '/home/shawnle/Documents/Restore_PoseCNN/PoseCNN-master/data_syn_LOV/data_2_objs/small'
    # data_path = '/home/shawnle/Documents/Projects/PoseCNN-master/data/LOV/3d_train_data/small'
    # data_path = 'D:\\SL\\3d_train_data\\small'
    data_path = '/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data_syn_LOV/data_2_objs'
    dat_gen = data_generator(data_path, num_classes=num_classes, batch_size=batch_size)

    # vgg16_weight_path = '.\\data\\imagenet_models\\vgg16.npy'
    vgg16_weight_path = './data/imagenet_models/vgg16.npy'
    sess = tf.Session()
    load(vgg16_weight_path, sess, ignore_missing=True)

    print('activation layer = ', md.layers[-4])
    print(md.layers[-1].shape)
    vertex_targets = tf.placeholder(tf.float32, shape=md.layers[-1].shape, name='vertex_targets')
    print("vtt shape = ", vertex_targets.shape)
    vertex_weights = tf.placeholder(tf.float32, shape=md.layers[-1].shape, name='vertex_weights')
    
    print(type(md.layers[-1]))
    print(type(vertex_targets))

    scores = md.layer_dict['prob']
    labels = md.layer_dict['gt_label_weight']
    loss_cls = md.loss_cross_entropy_single_frame(scores, labels)

    VERTEX_W = 5
    loss_vertex = VERTEX_W*md.smooth_l1_loss_vertex(md.layers[-1], vertex_targets, vertex_weights)

    # total_loss = mdw.smooth_l1_loss_vertex(ytrue, mdw.vertex_pred)
    # total_loss = md.smooth_l1_loss_vertex(md.layers[-1], vertex_targets, vertex_weights)
    total_loss = loss_cls + loss_vertex
    # optimizer = tf.train.MomentumOptimizer(0.0001, 0.9).minimize(total_loss)
    optimizer = tf.train.MomentumOptimizer(0.001, 0.9).minimize(total_loss)


    def unpad_im_tensor(im, factor):
        '''
            adapt from PCNN, applied to tensor
        '''
        # print('up im tensor im shape type =',im.get_shape(), type(im.get_shape()))
        height = im.get_shape().as_list()[1]  #im.shape[0]
        width = im.get_shape().as_list()[2] # im.shape[1]

        pad_height = int(np.ceil(height / float(factor)) * factor - height)
        pad_width = int(np.ceil(width / float(factor)) * factor - width)

        if len(im.shape) == 3:
            return im[0:height-pad_height, 0:width-pad_width, :]
        elif len(im.shape) == 2:
            return im[0:height-pad_height, 0:width-pad_width]


    ########### tensorboard reports
    with tf.name_scope('summaries'):
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.scalar('loss_cls', loss_cls)
        tf.summary.scalar('loss_vertex', loss_vertex)
        tf.summary.image('rgb_input', md.input)
        tf.summary.image('prob', scores)
    
        labels = md.layer_dict['label_2d']
        tf.summary.scalar('label_2d_sum_abs', tf.reduce_sum(tf.abs(labels)))
        labels_up = unpad_im_tensor(labels, 16)
        im_label = labels_to_image_tensor(labels_up)
        tf.summary.image('label_2d', tf.expand_dims(tf.cast(im_label, tf.uint8), 3))

        # tf.summary.image('label_2d', tf.expand_dims(tf.cast(md.layer_dict['label_2d'], tf.uint8), 3) )

        # labels = unpad_im(labels, 16)
        # im_label = labels_to_image(labels)


    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('.')
    writer.add_graph(tf.get_default_graph())
    writer.flush()

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    # config.gpu_options.allow_growth = True
    config.allow_soft_placement=True
    ########### tensorboard reports

    iter = 0
    total_step = int(1 / batch_size)
    print("total step per epoch = ", total_step)
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        if mode is 'test' :

            restore(sess, './output/vertex_pred/model.ckpt')

            print('load model successfully!')

            inp, out = get_a_sample(data_path, num_classes=3)

            feed_dict = { md.input : inp[0],
                          md.gt_label_2d : np.squeeze(inp[1], axis=0), 
                          vertex_targets: out[0],
                          vertex_weights: out[1] }

            summary, out_val, labels = sess.run([merged, md.output, md.layer_dict['label_2d']], feed_dict= feed_dict, options=run_options, run_metadata=run_metadata)

            print('amax rgb, lbl, tgt, w, est lb =', np.amax(inp[0]), np.amax(inp[1]), np.amax(out[0]), np.amax(out[1]), np.amax(labels))
            print('labels shape = ', labels, type(labels), labels.shape)#, tf.shape(labels[0][0][0]))
            imgplot = plt.imshow(labels[0,...])
            plt.show()
            print("finished.")

            exit()


        sess.run(tf.global_variables_initializer(), options=run_options, run_metadata=run_metadata)

        for epoch in range(5000):
            for step in range(total_step):
                inp, out = next(dat_gen)

                # print('len/shape inp/out: ', len(inp), len(out), inp[0].shape, out[0].shape, out[1].shape)
                if len(inp) == 0: 
                    print("empty data. Continue..")
                    continue

                feed_dict = { md.input : inp[0],
                              md.gt_label_2d : np.squeeze(inp[1], axis=0),
                              vertex_targets: out[0],
                              vertex_weights: out[1]
                }

                assert inp[0] is not None and np.squeeze(inp[1], axis=0) is not None and out[0] is not None and out[1] is not None

                _, summary, loss, mdinp, mdact, mdact1, out_val, labels = sess.run([optimizer, merged, total_loss, md.input, md.layers[8][0], md.layers[9][0], md.output, md.layer_dict['label_2d']], feed_dict= feed_dict, options=run_options, run_metadata=run_metadata)
                # _, loss, mdinp, mdact, mdact1, out_val, vertex_diff = sess.run([optimizer, total_loss, md.input, md.layers[8][0], md.layers[9][0], md.output, md.output -vertex_targets], feed_dict= feed_dict, options=run_options, run_metadata=run_metadata)

                # labels = unpad_im(labels, 16)
                # im_label = labels_to_image(labels)
                # cv2.imshow("im_label", im_label)
                # cv2.waitKey(0)

                writer.add_summary(summary, epoch)

                # np.save('mdinp_'+ str(iter) +'.npy', mdinp)
                # np.save('mdact_'+ str(iter) +'.npy', mdact)
                # np.save('mdact1_'+ str(iter) +'.npy', mdact1)
                # np.save('out_val_'+ str(iter) +'.npy', out_val)
                # np.save('vertex_diff_' + str(iter) + '.npy', vertex_diff)

                print('iter ' + str(iter) + '/' + str(epoch) + ': --> loss:', loss)
                iter = iter + 1

                if iter % 200 == 0:
                    save_path = saver.save(sess, "./output/vertex_pred/model.ckpt")
                    print("Model saved in path: %s" % save_path)

