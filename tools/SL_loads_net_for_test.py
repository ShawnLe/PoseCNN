
import _init_paths

from easydict import EasyDict as edict
import cv2
import tensorflow as tf

from fcn.config import cfg, cfg_from_file, get_output_dir
from networks.factory import get_network
from datasets.factory import get_imdb
from fcn.train import get_training_roidb
from fcn.train import loss_cross_entropy_single_frame, smooth_l1_loss_vertex, load_and_enqueue
from gt_synthesize_layer.layer import GtSynthesizeLayer
from gt_data_layer.layer import GtDataLayer

class input_args(object):
    
    def __init__(self):
        self.config_file = "../experiments/cfgs/lov_color_2d.yml"
        # self.network_name = "vgg16"
        self.network_name = "vgg16_convs"
        self.is_train = False
        self.trained_model = "../output/lov/lov_train/vgg16_fcn_color_single_frame_2d_pose_add_lov_iter_3000.ckpt"
        # self.imdb_name = "lov_keyframe"
        self.imdb_name = "lov_train"
        self.CAD = "data/LOV/models.txt"
        self.POSE = "data/LOV/poses.txt"

class loss_definition(object):

    def __init__(self, network, cfg):
        
        self.loss_regu = tf.add_n(tf.losses.get_regularization_losses(), 'regu')

        scores = network.get_output('prob')
        labels = network.get_output('gt_label_weight')
        self.loss_cls = loss_cross_entropy_single_frame(scores, labels)

        vertex_pred = network.get_output('vertex_pred')
        vertex_targets = network.get_output('vertex_targets')
        vertex_weights = network.get_output('vertex_weights')
        # loss_vertex = tf.div( tf.reduce_sum(tf.multiply(vertex_weights, tf.abs(tf.subtract(vertex_pred, vertex_targets)))), tf.reduce_sum(vertex_weights) + 1e-10 )
        self.loss_vertex = cfg.TRAIN.VERTEX_W * smooth_l1_loss_vertex(vertex_pred, vertex_targets, vertex_weights)

        self.loss_pose = cfg.TRAIN.POSE_W * network.get_output('loss_pose')[0]

        self.loss = self.loss_cls + self.loss_vertex + self.loss_pose + self.loss_regu


import datasets.lov
import numpy as np
class sub_factory(object):
    """ only 2 datasets are supported for now """
    def __init__(self):

        self.__sets = {}

        # lov dataset
        for split in ['train', 'keyframe']:
            name = 'lov_{}'.format(split)
            print name
            self.__sets[name] = (lambda split=split:
                    datasets.lov(split))

    def get_imdb(self, name):
        """Get an imdb (image database) by name."""
        if not self.__sets.has_key(name):
            raise KeyError('Unknown dataset: {}'.format(name))
        return self.__sets[name]()


# **** get configs
args = input_args()    

cfg_from_file(args.config_file)
cfg.IS_TRAIN = args.is_train

print cfg
print '** current run mode'
print 'cfg.TRAIN.SINGLE_FRAME = ' + str(cfg.TRAIN.SINGLE_FRAME)
print 'cfg.TRAIN.VERTEX_REG_2D = ' + str(cfg.TRAIN.VERTEX_REG_2D)
print 'cfg.TRAIN.VERTEX_REG_3D = ' + str(cfg.TRAIN.VERTEX_REG_3D)
print 'cfg.TRAIN.POSE_REG = ' + str(cfg.TRAIN.POSE_REG)
print 'cfg.TRAIN.ADAPT = ' + str(cfg.TRAIN.ADAPT)
print 'cfg.INPUT = ' + str(cfg.INPUT)
print 'cfg.TEST.SINGLE_FRAME = ' + str(cfg.TEST.SINGLE_FRAME)
print 'cfg.TEST.SEGMENTATION = ' + str(cfg.TEST.SEGMENTATION)

assert cfg.TRAIN.SINGLE_FRAME == True
assert cfg.TRAIN.VERTEX_REG_2D == True
assert cfg.TRAIN.VERTEX_REG_3D == False
assert cfg.TRAIN.POSE_REG == True
assert cfg.TRAIN.ADAPT == False
assert cfg.INPUT == 'COLOR'
assert cfg.TEST.SEGMENTATION == True
assert cfg.TEST.SINGLE_FRAME == True

# **** get network
network = get_network(args.network_name)
print 'Use network `{:s}` in training'.format(args.network_name)

# **** define losses
losses = loss_definition(network, cfg)

# **** load data
sub_fact = sub_factory()
imdb = sub_fact.get_imdb(args.imdb_name)
roidb = get_training_roidb(imdb)
# data_layer = GtSynthesizeLayer(roidb, imdb.num_classes, imdb._extents, imdb._points_all, imdb._symmetry, imdb.cache_path, imdb.name, imdb.data_queue, cfg.CAD, cfg.POSE)
data_layer = GtDataLayer(roidb, imdb.num_classes)

# **** load the trained model
print '** try to restore trained weights'
# tf.reset_default_graph()
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, args.trained_model)
    print ("model restored.")

    coord = tf.train.Coordinator()
    load_and_enqueue(sess, network, data_layer, coord, 0)

    sess.run([loss])


