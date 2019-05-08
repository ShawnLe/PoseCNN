from __future__ import print_function

import glob
import os

from math import ceil
import numpy as np
from scipy.io import loadmat

import cv2

# import keras
from tensorflow.python import keras

# from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Add, Concatenate, Dropout, Dense, Lambda
from tensorflow.python.keras.models import Model, load_model
# from tensorflow.python.keras.models import load_model

# from keras.layers import Concatenate, Dropout, Dense, Lambda
# from keras.models import Model
# from keras.models import load_model

from tensorflow.python.keras import backend as K
# import keras.backend as K
from tensorflow.python.keras.layers import Layer
# import tensorflow.python.keras.layers as KL
# import tensorflow.python.keras.engine as KE
# from keras import backend as K
# from keras.layers import Layer
# import keras.layers as KL
# import keras.engine as KE

import tensorflow as tf

from numpy import linalg as LA

############################################################
#  Network Class
############################################################

class vgg16convs_vertex_pred():

    def __init__(self, input):

        self.build(input)

    def backend_debug_print(self, x):
        K.print_tensor(x, message='hello_print') # , [tf.shape(x)]
        return x

    def build(self, input):

        # # 1st block
        # conv1_1 = Conv2D(64, (3,3), name='conv1_1', padding='same', activation='relu')(input)
        # conv1_2 = Conv2D(64, (3,3), name='conv1_2', padding='same', activation='relu')(conv1_1)
        # pool1 = MaxPooling2D((2,2), strides=(2,2), name='pool1')(conv1_2)

        # # 2nd block
        # conv2_1 = Conv2D(128, (3,3), name='conv2_1', padding='same', activation='relu')(pool1)
        # conv2_2 = Conv2D(128, (3,3), name='conv2_2', padding='same', activation='relu')(conv2_1)
        # pool2 = MaxPooling2D((2,2), strides=(2,2), name='pool2')(conv2_2)

        # # 3rd block
        # conv3_1 = Conv2D(256, (3,3), name='conv3_1', padding='same', activation='relu')(pool2)
        # conv3_2 = Conv2D(256, (3,3), name='conv3_2', padding='same', activation='relu')(conv3_1)
        # conv3_3 = Conv2D(256, (3,3), name='conv3_3', padding='same', activation='relu')(conv3_2)
        # pool3 = MaxPooling2D((2,2), strides=(2,2), name='pool3')(conv3_3)

        # # 4th block
        # conv4_1 = Conv2D(512, (3,3), name='conv4_1', padding='same', activation='relu')(pool3)
        # conv4_2 = Conv2D(512, (3,3), name='conv4_2', padding='same', activation='relu')(conv4_1)
        # conv4_3 = Conv2D(512, (3,3), name='conv4_3', padding='same', activation='relu')(conv4_2)
        # pool4 = MaxPooling2D((2,2), strides=(2,2), name='pool4')(conv4_3)

        # # 5th block
        # conv5_1 = Conv2D(512, (3,3), name='conv5_1', padding='same', activation='relu')(pool4)
        # conv5_2 = Conv2D(512, (3,3), name='conv5_2', padding='same', activation='relu')(conv5_1)
        # conv5_3 = Conv2D(512, (3,3), name='conv5_3', padding='same', activation='relu')(conv5_2)

        vgg16 = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input)
        print('vgg output :', vgg16.output.get_shape())
        # until the last 2 layers, all are freezed for training
        for layer in vgg16.layers:
          layer.trainable = False

        conv4_3 = vgg16.layers[-6].output
        conv5_3 = vgg16.layers[-2].output # 2nd layer from the last, block5_conv3

        # vertex pred head
        num_units = 64
        keep_prob_queue = 0.5 # YX uses keep_prob
        rate = 1 - keep_prob_queue # keras uses rate
        scale = 1.

        score_conv5 = Conv2D(num_units, (1,1), name='score_conv5', padding='same', activation='relu')(conv5_3)
        # upscore_conv5 = deconv(score_conv5, 4, 4, num_units, 2, 2, name='upscore_conv5', trainable=False)  # how to make a keras equivalent layer?
        # Keras bug: before and after KL.Conv2DTranspose, shape is lost. Use tensorflow keras instead. Link: https://github.com/keras-team/keras/issues/6777
        # score_conv5: (?, 30, 40, 64)
        # upscore_conv5: (?, ?, ?, 64)
        print('score_conv5:', score_conv5.shape)
        upscore_conv5 = Conv2DTranspose(num_units, (4,4), strides=(2,2), name='upscore_conv5', padding='same', data_format="channels_last", trainable=False)(score_conv5)
        print('upscore_conv5:', upscore_conv5.shape)

        score_conv4 = Conv2D(num_units, (1,1), name='score_conv4', padding='same', activation='relu')(conv4_3)

        # score_conv4_p = Lambda(self.backend_debug_print)(score_conv4)  # , output_shape=K.shape(score_conv4)
        score_conv4_p = K.print_tensor(score_conv4)

        # add_score = Add()([score_conv4, upscore_conv5])
        add_score = Add()([score_conv4_p, upscore_conv5])

        dropout = Dropout(rate, name='dropout')(add_score)
        
        #upscore = deconv(dropout, int(16*scale), int(16*scale), num_units, int(8*scale), int(8*scale), name='upscore', trainable=False)
        upscore = Conv2DTranspose(num_units, (int(16*scale), int(16*scale)), strides=(int(8*scale), int(8*scale)), name='upscore', padding='same', data_format="channels_last", trainable=False)(dropout)
        print('output shape: ', upscore.get_shape())

        # 'prob' and 'label_2d' will be added later. 'gt_label_weight' cannot be added because of hard_label C++ module

        vertex_reg = 1
        num_classes = 3
        if vertex_reg:
            init_weights = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001)
            init_bias = keras.initializers.Constant(value=0.)
            regularizer = keras.regularizers.l2(0.0001)

            # score_conv5_vertex = Conv2D(128, (1,1), name='score_conv5_vertex', padding='same', activation='relu')(conv5_3)
            score_conv5_vertex = Conv2D(128, (1,1), name='score_conv5_vertex', padding='same', activation='relu', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(conv5_3)
            #   upscore_conv5_vertex = deconv(score_conv5_vertex, 4, 4, 128, 2, 2, name='upscore_conv5_vertex')
            upscore_conv5_vertex = Conv2DTranspose(128, (4, 4), strides=(2, 2), name='upscore_conv5_vertex', padding='same', data_format="channels_last", trainable=False)(score_conv5_vertex)

            
            # score_conv4_vertex = Conv2D(128, (1,1), name='score_conv4_vertex', padding='same', activation='relu')(conv4_3)
            score_conv4_vertex = Conv2D(128, (1,1), name='score_conv4_vertex', padding='same', activation='relu', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(conv4_3)
            
            add_score_vertex = Add()([score_conv4_vertex, upscore_conv5_vertex])
            dropout_vertex = Dropout(rate, name='dropout_vertex')(add_score_vertex)
            #   upscore_vertex = deconv(dropout_vertex, int(16*scale), int(16*scale), 128, int(8*scale), int(8*scale), name='upscore_vertex', trainable=False)
            upscore_vertex = Conv2DTranspose(128, (int(16*scale), int(16*scale)), strides=(int(8*scale), int(8*scale)), name='upscore_vertex', padding='same', data_format="channels_last", trainable=False)(dropout_vertex)
            
            # 3*num_classes == depth == # channels-> a fixed output like this 
            # vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu')(upscore_vertex)
            vertex_pred = Conv2D(3*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu', kernel_initializer=init_weights, bias_initializer=init_bias, kernel_regularizer=regularizer, bias_regularizer=regularizer)(upscore_vertex)
            # vertex_pred = Conv2D(2*num_classes, (1,1), name='vertex_pred', padding='same', activation='relu')(upscore_vertex)
            # vertex_pred = K.reshape(vertex_pred, (480, 640, 3*num_classes))


        # create keras model
        self.the_model = Model(inputs=input, outputs=vertex_pred)


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

    # print("[_vote_centers] hello.")
    # print("center = ", center)

    width = im_label.shape[1]
    height = im_label.shape[0]
    vertex_targets = np.zeros((height, width, 3*num_classes), dtype=np.float32)
    vertex_weights = np.zeros(vertex_targets.shape, dtype=np.float32)

    c = np.zeros((2, 1), dtype=np.float32)
    for i in xrange(1, num_classes):
        y, x = np.where(im_label == i)
        if len(x) > 0:
            # print("I am working. len(x) = ", len(x))

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

    # np.save("vx.npy", vertex_targets[:,:,3])
    # cv2.imwrite("vx.png", vertex_targets[:,:,3])

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

        b = b+1

        if b >= batch_size:
            inputs = [batch_rgb]
            outputs = [batch_center]

            b = 0

            yield (inputs, outputs)
        

def pixel_voting(batch_center):
    '''
        a quick prototype for pixel-based voting of keypoints to be, later, become a tensor op
    '''

    center_targets = batch_center[0,:,:,:]
    print('center_targets shape =', center_targets.shape)

    dims = center_targets.shape[0:2]
    print (dims)

    cv2.imwrite('vx.png', center_targets[:,:,0])
    np.save('vx.npy', center_targets[:,:,0])

    # do the voting
    vote_space = np.zeros(shape=(dims[0], dims[1]))
    for y in xrange(dims[0]):
        for x in xrange(dims[1]):

            ux = center_targets[y,x,0]
            uy = center_targets[y,x,1]

            norm = LA.norm([ux, uy])

            if (norm > 0):
                # print("norm > 0")
                ux_ = ux / norm
                uy_ = uy / norm

                # delta = uy_ / ux_

                xi = x
                yi = y
                # if ux_ >= 0:
                #     inc = 1.
                # else:
                #     inc = -1.

                while 0<=xi and xi<dims[1] and 0<=yi and yi<dims[0]:

                    vote_space[int(yi), int(xi)] = vote_space[int(yi), int(xi)] + 1.

                    # xi = xi+inc
                    # yi = delta * (xi - x) + y

                    xi = xi + ux_
                    yi = yi + uy_

    cv2.imwrite('vote_space.png', vote_space)
    np.save("vote_space.npy", vote_space)

    return None

    
############################################################
#  Main
############################################################

if __name__ == "__main__":

    # execute only if run as a script
    # main()

    input = Input(shape=(480, 640, 3), dtype='float32')

    mdw = vgg16convs_vertex_pred(input)
    # print('model output shape {}'.format(mdw.the_model.output_shape))

    # rmsprop = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
    sgd = keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=0.9)

    mdw.the_model.compile(optimizer=sgd,
                loss='mean_squared_error',
                metrics=['accuracy'])

    # mdw.the_model.summary()                  

    optim = mdw.the_model.optimizer
    ws = []
    grads = []
    grad_funs = []
    # print("mdw.the_model.input = {}".format(mdw.the_model.input))
    # print("mdw.the_model.output = {}".format(mdw.the_model.layers[-1].output))
    # print("targets[0] = {}".format(mdw.the_model.targets[0]))
    # input_tensors = [mdw.the_model.input[0],
    #                 #  mdw.the_model.sample_weights[0],
    #                  K.placeholder(shape=(None,None,None,9)),
    #                  K.learning_phase()]
    # print("the_model.sample_weights = {}".format(mdw.the_model.sample_weights[0]))
    # input_tensors = [K.placeholder(shape=(None, 480, 640, 3), dtype='float32'),
    #                 # K.placeholder(shape=(480, 640, 3), dtype='float32'),  # input shape
    #                 # K.placeholder(shape=(None,None,None,9)), # output shape
    #                 # K.placeholder(shape=(None)),

    #                 mdw.the_model.sample_weights[0],
    #                 # K.placeholder([3]),
                    
    #                 # K.placeholder(shape=(480,640,9)),
    #                 mdw.the_model.targets[0],
    #                 K.learning_phase()]

    # print(input_tensors)

    weights = mdw.the_model.trainable_weights
    print("len(weights)", len(weights))
    # print("weights:")
    # for weight in weights:
    #     print(weight)
        # print (K.is_keras_tensor(weight))
        # print ("w name ={}".format(weight.name[:-2]))
    #     if mdw.the_model.get_layer(weight.name[:-2]).trainable:
    #         ws.append(weight)

    # weights = [weight for weight in weights if mdw.the_model.get_layer(weight.name[:-2]).trainable] 
    # get_gradients returns tensors
    print("tt loss tensor {}".format(mdw.the_model.total_loss))
    gradients = optim.get_gradients(mdw.the_model.total_loss, weights)

    print (gradients)
    # print (gradients[0].op)
    # print (gradients[0].graph)
    # print ("gradients is keras tensor? {}".format(K.is_keras_tensor(gradients[0])))

    # get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    # print(get_gradients)

    # for w in mdw.the_model.trainable_weights:
    #     ws.append(w)
    #     grad = optim.get_gradients(mdw.the_model.total_loss, w)
    #     grads.append(grad)
    #     print(w)
    #     print(grad)
    #     get_gradients = K.function(inputs=input_tensors, outputs=grad)

    num_classes = 3 # including the background as '0'

    # data_path = '/home/shawnle/Documents/Restore_PoseCNN/PoseCNN-master/data_syn_LOV/data_2_objs/'
    data_path = '/home/shawnle/Documents/Projects/PoseCNN-master/data/LOV/3d_train_data'
    dat_gen = data_generator(data_path, num_classes=num_classes)

    # mode = 'INFERENCE'
    mode = 'TRAIN'
    
    # test_model = load_model('my_model.h5')

    # print("input = {}".format(test_model.inputs))
    # print("output = {}".format(test_model.outputs))

    a_sample = get_a_sample(data_path, num_classes=num_classes)
    print(np.shape(a_sample[0]))
    print(np.shape(a_sample[1]))

    # np.save("vx.npy", a_sample[1][0,:,:,3])
    # cv2.imwrite("vx.png", a_sample[1][0,:,:,3])
    # print("vertext_pred_target = {}".format(test_model.targets[0].name))
    # print("vertext_pred_target len = {}".format(len(test_model.targets)))
    # print("vertext_pred_target shape = {}".format(test_model.targets[0].shape))

    # g = tf.get_default_graph()
    # print("all ops = {}".format(g.get_operations()))

    cls_id = 1
    pixel_voting(a_sample[1][:,:,:, 3*cls_id : 3*cls_id +3])
    