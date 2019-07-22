#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a FCN on an image database."""

import _init_paths
import argparse
import os, sys
from transforms3d.quaternions import quat2mat
from fcn.config import cfg, cfg_from_file, get_output_dir
import libsynthesizer
import cPickle

from datasets.factory import get_imdb
import numpy as np
import numpy.random as npr
from numpy import linalg as LA

import scipy.io
import cv2
import numpy as np

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--ckpt', dest='pretrained_ckpt',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--rig', dest='rig_name',
                        help='name of the camera rig file',
                        default=None, type=str)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args



# def vertice_is_good(kpt_num, selMdlPoints, points, class_num, sel_num, vt_id):

#     ret = True
#     dist_thres = .020
    
#     current_point =  points[class_num, vt_id, :]

#     for i in xrange(kpt_num*class_num, kpt_num*class_num +sel_num):
#         chosen_point = selMdlPoints[:,i]

#         if LA.norm(chosen_point -current_point) < dist_thres:
#             ret = False
#             break
#     return ret


# """ @brief choose a random set of keypoints for each model
#     @param[out] selMdlPoints
# """
# def selectModelPoints(num_classes, kpt_num, points):

#     selMdlPoints = np.zeros([3, num_classes * kpt_num], dtype=np.float32)

#     vertice_idx = 0
#     for i in xrange(num_classes):
#         for j in xrange(kpt_num):
#             while not vertice_is_good(kpt_num, selMdlPoints, points, i, j, vertice_idx): 
#                 vertice_idx = npr.randint(0, points.shape[1])
#             selMdlPoints[:, i*kpt_num + j]  = points[i, vertice_idx, :]
#             # print 'points[i, vertice_idx, :] =' + str( points[i, vertice_idx, :])
#             # print selMdlPoints[:, i*kpt_num + j]

#     return selMdlPoints

# """ @brief check if the projected point is visible or not
#     @param[out] visibility status {0: out-of-FOV, 1: hidden, 2: visible}
# """
# def checkVisibility(width, height, P3d, p2d, depthmap):

#     p2d_ = p2d

#     p2d = np.floor(p2d).astype(dtype=np.int)
#     if p2d[0] < 0 or width <= p2d[0]  \
#        or p2d[1] < 0 or height <= p2d[1]:
#         vis = 0
#     else:
#         # Z_depthmap = depthmap[p2d[1], p2d[0]] # need to use sub-pixel accessing
#         # print 'rounding depthmap'
#         # print depthmap[p2d[1], p2d[0]]
#         # print depthmap[p2d[1]+1, p2d[0]]
#         # print depthmap[p2d[1], p2d[0]+1]
#         # print depthmap[p2d[1]+1, p2d[0]+1]

#         p2d_ = np.reshape(p2d_,(1,1,2)).astype(dtype=np.float32)
#         # print 'p2d_ = ' + str(p2d_)        
#         # print 'p2d_ shape = ' + str(p2d_.shape)
    
#         Z_depthmap = cv2.remap(depthmap, p2d_, None, cv2.INTER_LINEAR)
#         # print 'sub-pixel = ' + str(Z_depthmap)

#         #if Z_depthmap < P3d[2]:
#         if P3d[2] - Z_depthmap > .001 :
#             vis = 1
#         else:
#             vis = 2

#     return vis


def load_object_extents(data_path, num_classes):

    extent_file = os.path.join(data_path, 'LOV', 'extents.txt')
    assert os.path.exists(extent_file), \
            'Path does not exist: {}'.format(extent_file)

    extents = np.zeros((num_classes, 3), dtype=np.float32)
    # extents[1:, :] = np.loadtxt(extent_file)
    extents = np.loadtxt(extent_file)

    return extents

def _get_bb3D(extent):
    bb = np.zeros((3, 8), dtype=np.float32)

    xHalf = extent[0] * 0.5
    yHalf = extent[1] * 0.5
    zHalf = extent[2] * 0.5

    bb[:, 0] = [xHalf, yHalf, zHalf]
    bb[:, 1] = [-xHalf, yHalf, zHalf]
    bb[:, 2] = [xHalf, -yHalf, zHalf]
    bb[:, 3] = [-xHalf, -yHalf, zHalf]
    bb[:, 4] = [xHalf, yHalf, -zHalf]
    bb[:, 5] = [-xHalf, yHalf, -zHalf]
    bb[:, 6] = [xHalf, -yHalf, -zHalf]
    bb[:, 7] = [-xHalf, -yHalf, -zHalf]

    return bb

def is_qualified(pose, extent, intrinsic_matrix, height, width):
    """ check whether pose gives whole bounding-box inside FOV
    """
    # print(pose)
    # print(extent)

    bb3D = _get_bb3D(extent)
    # print(bb3D)
    RotationMatrix = pose[:,:3]
    # print(RotationMatrix)
    # print(RotationMatrix.shape)
    TranslationMatrix = pose[:,3].reshape(3,1)
    # print(TranslationMatrix)
    # print(TranslationMatrix.shape)
    coor2d = np.matmul(intrinsic_matrix, np.matmul(RotationMatrix, bb3D) + TranslationMatrix)
    x_coor = coor2d[0] / coor2d[2]
    y_coor = coor2d[1] / coor2d[2]
    # print (x_coor)
    # print (y_coor)
    # print (len(x_coor))

    for i in range(len(x_coor)):
        if x_coor[i] < 0 or x_coor[i] >= width or y_coor[i] < 0 or y_coor[i] >= height:
            return False

    return True


if __name__ == '__main__':

    import json

    myfile = os.path.join(os.getcwd(), 'lib','synthesize_4Y2', 'syn_cfg.json')
    print "Opening ", myfile

    with open(myfile, 'r') as f:
        syn_cfg=json.load(f)

    print (syn_cfg["intrinsics"]["fx"])
    # exit()

    args = parse_args()
    cfg.BACKGROUND = args.background_name

    print 'imdb_name = ' + args.imdb_name
    imdb = get_imdb(args.imdb_name)
    print (imdb.num_classes)

    # check number of instance
    num_instances = syn_cfg["num_instances"]
    assert len(num_instances) == imdb.num_classes -1 , "len(num_instance)=" + str(len(num_instances)) + " && num_classes-1=" + str(imdb.num_classes-1)
    num_instances_ = np.array(num_instances, dtype=np.float32)


    extents = load_object_extents(os.path.join(os.getcwd(),'data'), imdb.num_classes-1)
    print ("extents = %s" % (extents))
    print ("extents shape = {}" .format(extents.shape))

    # num_images = 500
    # num_images = 2000
    # height = 480
    # width = 640
    # fx = 1066.778
    # fy = 1067.487
    # px = 312.9869
    # py = 241.3109

    num_images = syn_cfg["num_images"]
    height = syn_cfg["height"]
    width = syn_cfg["width"]
    fx = syn_cfg["intrinsics"]["fx"]
    fy = syn_cfg["intrinsics"]["fy"]
    px = syn_cfg["intrinsics"]["px"]
    py = syn_cfg["intrinsics"]["py"]
    zfar = 6.0
    znear = 0.25
    tnear = 0.5
    tfar = 1.5
    num_classes = imdb.num_classes -1
    factor_depth = syn_cfg["factor_depth"] # 10000.0
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])
    # root = '/capri/YCB_Video_Dataset/data_syn/'
    root = '/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data_syn_LOV/'
    # root = '/home/shawnle/Documents/Projects/PoseCNN-master/data/LOV_syn/'

    if not os.path.exists(root):
        os.makedirs(root)

    synthesizer_ = libsynthesizer.Synthesizer(args.cad_name, args.pose_name)
    # synthesizer_.setup(width, height)
    synthesizer_.setup_multi_inst(width, height, num_instances_)
    synthesizer_.init_rand(1200)

    parameters = np.zeros((8, ), dtype=np.float32)
    parameters[0] = fx
    parameters[1] = fy
    parameters[2] = px
    parameters[3] = py
    parameters[4] = znear
    parameters[5] = zfar
    parameters[6] = tnear
    parameters[7] = tfar

    # load background
    cache_file = cfg.BACKGROUND
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            backgrounds = cPickle.load(fid)
        print 'backgrounds loaded from {}'.format(cache_file)

    i = 0
    points = imdb._points_all
    # num_kpt = 20
    num_kpt = syn_cfg["num_keypoints"]
    num_cls = points.shape[0]-1

    if syn_cfg["sel_random_points"]:
        selMdlPoints = selectModelPoints(num_cls, num_kpt, points[1:,:,:])  # not use the background

    else:
        selMdlPoints = syn_cfg["selectModelPoints"]

        print (selMdlPoints)
        print (np.array(selMdlPoints).shape)
        shape = np.array(selMdlPoints).shape

        assert (shape[1] == 3) #, "current shape[1] = {}", (shape[1])
        assert (shape[0] == num_cls * num_kpt)

        selMdlPoints = np.array(selMdlPoints)
        selMdlPoints = np.transpose(selMdlPoints) #reshape((3, num_cls * num_kpt))

    print ('selMdlPoints = ', selMdlPoints)
    print ('selMdlPoints.shape = ', selMdlPoints.shape)
    print("finishes selecting random points.")

    while i < num_images:

        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        depth_syn = np.zeros((height, width, 3), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        # class_indexes = -1 * np.ones((num_classes, ), dtype=np.float32)
        # poses = np.zeros((num_classes, 7), dtype=np.float32)
        # centers = np.zeros((num_classes, 2), dtype=np.float32)
        class_indexes = -1 * np.ones((np.sum(num_instances), ), dtype=np.float32)
        poses = np.zeros((np.sum(num_instances), 7), dtype=np.float32)
        centers = np.zeros((np.sum(num_instances), 2), dtype=np.float32)
        # num_instances_ = np.array(num_instances, dtype=np.float32)
        is_sampling = True
        is_sampling_pose = False

        synthesizer_.render_python(int(width), int(height), parameters, num_instances_, \
                                   im_syn, depth_syn, vertmap_syn, class_indexes, poses, centers, is_sampling, is_sampling_pose)

        # print('poses =', poses)
        # print('poses shape =', poses.shape)
        # print('poses[0,:]', poses[0,:])
        # print('poses[1,:]', poses[1,:])
        # print('poses[2,:]', poses[2,:])
        # exit()  

        # convert images
        im_syn = np.clip(255 * im_syn, 0, 255)
        im_syn = im_syn.astype(np.uint8)
        depth_syn = depth_syn[:, :, 0]

        # convert depth
        im_depth_raw = factor_depth * 2 * zfar * znear / (zfar + znear - (zfar - znear) * (2 * depth_syn - 1))
        I = np.where(depth_syn == 1)
        im_depth_raw[I[0], I[1]] = 0

        # compute labels from vertmap
        label = np.round(vertmap_syn[:, :, 0]) + 1
        label[np.isnan(label)] = 0

        # convert pose
        index = np.where(class_indexes >= 0)[0]
        print("class_indexes=", class_indexes)
        print("index=",index)
        num = num_classes #len(index)
        sum_num_inst = np.sum(num_instances)
        # qt = np.zeros((3, 4, num), dtype=np.float32)
        qt = np.zeros((3, 4, sum_num_inst), dtype=np.float32)

        # whole set of poses, each has whole bounding-box inside FOV
        set_is_qualified = True
        inst = 0
        for j in xrange(num):
            for k in xrange(num_instances[j]):
                ind = index[j]
                qt[:, :3, inst] = quat2mat(poses[inst, :4])
                qt[:, 3, inst] = poses[inst, 4:]

                inst = inst + 1

                if syn_cfg["check_whole_bb"]:
                    if not is_qualified(qt[:,:,inst], extents[j,:], intrinsic_matrix, height, width):
                        set_is_qualified = False
                        break

        if not set_is_qualified:
            continue

        # print('qt[0] =', qt[:,:,0])
        # print('qt[0] shape =', qt[:,:,0].shape)
        # print('qt[1] =', qt[:,:,1])

        # if the projected blob of object is too small, then skip it
        flag = 1
        for j in xrange(num):
            cls = index[j] + 1
            I = np.where(label == cls)
            if len(I[0]) < 800*num_instances[index[j]]:
                flag = 0
                break
        if flag == 0:
            continue

        # process the vertmap
        vertmap_syn[:, :, 0] = vertmap_syn[:, :, 0] - np.round(vertmap_syn[:, :, 0])
        vertmap_syn[np.isnan(vertmap_syn)] = 0

        # metadata
        metadata = {'poses': qt.tolist(), 'cls_indexes': (class_indexes[index].astype(int) + 1).tolist()}

        # sample a background image
        rgba = im_syn

        if syn_cfg["random_background"]:
            
            ind = np.random.randint(len(backgrounds), size=1)[0]
            filename = backgrounds[ind]
            print("background is %d named %s chosen" % (ind, filename))                        
        else:
            ind = syn_cfg["background_id"]
            filename = backgrounds[ind]
            print("background %d named %s is chosen" % (ind, filename))
            
            # filename = backgrounds[10]

        background = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        try:
            background = cv2.resize(background, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)
        except:
            if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'NORMAL':
                background = np.zeros((rgba.shape[0], rgba.shape[1]), dtype=np.uint16)
            else:
                background = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
            print 'bad background image'

        if cfg.INPUT != 'DEPTH' and cfg.INPUT != 'NORMAL' and len(background.shape) != 3:
            background = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
            print 'bad background image'

        # add background
        im = np.copy(rgba[:,:,:3])
        alpha = rgba[:,:,3]
        I = np.where(alpha == 0)
        if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'NORMAL':
            im_depth[I[0],I[1]] = background[I[0], I[1]] / 10
        else:
            im[I[0], I[1], :] = background[I[0], I[1], :3]

        # sampling and project model points
        Y2_meta = []
        # points = imdb._points_all
        # num_kpt = 20
        # num_cls = points.shape[0]-1
        # selMdlPoints = selectModelPoints(num_cls, num_kpt, points[1:,:,:])  # not use the background

        im_test = np.array(im, copy=True)
        inst = 0
        for id in xrange(num_cls):
            for jj in xrange(num_instances[id]):

                inst_rec = []
                RT = qt[:,:,inst]

                P3d = np.ones((4, num_kpt), dtype=np.float32)
                P3d[:3,:] = selMdlPoints[:, id*num_kpt : (id+1)*num_kpt]

                P3d_c = np.matmul(RT, P3d)
                p2d = np.matmul(intrinsic_matrix, P3d_c)
                p2d[0, :] = np.divide(p2d[0, :], p2d[2, :])
                p2d[1, :] = np.divide(p2d[1, :], p2d[2, :])

                for ii in xrange(p2d.shape[1]):

                    p = p2d[:2,ii]
                    status = checkVisibility(width, height, P3d_c[:,ii], p, im_depth_raw / factor_depth)
                    # Y2_meta.append( (p2d[0,ii], p2d[1,ii], status, id, inst) )
                    inst_rec.append( (p2d[0,ii], p2d[1,ii], status, id, inst) )

                    if status == 2:
                        p = p.astype(dtype=np.int)
                        cv2.circle(im_test, (p[0], p[1]), 3, imdb._class_colors[id], -1)
                    
                Y2_meta.append(inst_rec)
                inst = inst + 1

        # save 4Y2 meta
        # filename = root + '{:06d}-cpm_meta.txt'.format(i)      
        # f = open(filename, 'w')
        # for l in xrange(len(Y2_meta)):
        #     f.write(str(Y2_meta[l][0]) + ' ' + str(Y2_meta[l][1]) + ' ' + str(Y2_meta[l][2]) + ' ' + str(Y2_meta[l][3]) + ' ' + str(Y2_meta[l][4]) + '\n')
        # f.close()
        metadata['cpm_meta'] = Y2_meta

        # visibility verification
        filename = root + '{:06d}-projection.png'.format(i)      
        cv2.imwrite(filename, im_test)
        print filename

        # save image
        filename = root + '{:06d}-color.png'.format(i)
        # cv2.imwrite(filename, im_syn)
        cv2.imwrite(filename, im)
        print filename

        # save depth
        filename = root + '{:06d}-depth.png'.format(i)
        cv2.imwrite(filename, im_depth_raw.astype(np.uint16))

        # save label
        filename = root + '{:06d}-label.png'.format(i)
        cv2.imwrite(filename, label.astype(np.uint8))

        # save meta_data
        filename = root + '{:06d}-meta.json'.format(i)
        # scipy.io.savemat(filename, metadata, do_compression=True)
        with open(filename, 'w') as fp:
            json.dump(metadata, fp)
        # print(metadata)

        i += 1