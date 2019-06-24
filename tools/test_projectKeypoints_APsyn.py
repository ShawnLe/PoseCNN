'''
@brief to read old exported PCNN real data and merge it with synthesize data, finally export to train iPbNet workflow
@details general workflow:
        read real data -> for each frame:
            + synthesize a frame
            + merge synthesized frame with real frame (used as background)
            + project keypoints on the merged frame
            + modify and export meta data
@author shawnle
'''
import os, sys
import os.path as osp
import cPickle
import json

import numpy as np
import numpy.linalg as la
from transforms3d.quaternions import quat2mat
import cv2

import libsynthesizer

import _init_paths
from test_synthesis_4Y2 import selectModelPoints, checkVisibility, selectModelPointsBySpatialProp, calc_target_spatial

from datasets.factory import get_imdb

class config():
    def __init__(self):
        self.imdb_name = 'lov_keyframe'
        self.BACKGROUND = 'data/cache/backgrounds.pkl'
        self.data_num = 222
        self.root = '/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data_syn_LOV/'


def prepare_augment_from_real_data(mat_file):
    '''
    some description
    '''

    def prepare_from_json(imdb):

        myfile = os.path.join(os.getcwd(), 'lib','synthesize_4Y2', 'syn_cfg.json')
        print "Opening ", myfile
        with open(myfile, 'r') as f:
            syn_cfg=json.load(f)

        # check number of instance
        num_instances = syn_cfg["num_instances"]
        # assert len(num_instances) == imdb.num_classes -1 , "len(num_instance)=" + str(len(num_instances)) + " && num_classes-1=" + str(imdb.num_classes-1)
        num_instances_ = np.array(num_instances, dtype=np.float32)

        num_kpt = syn_cfg["num_keypoints"]       

        return {'num_instances' : num_instances_,
                'num_kpt' : num_kpt,
                'syn_cfg' : syn_cfg}


    def calc_spatial_prop(points):
        '''
        @note: spatial_props[id] refers to object class (id+1)-th
        '''

        num = points.shape[0]
        spatial_props = []
        for cls_id in range(1,num):

            class_pts = points[cls_id,...].T

            covar = np.cov(class_pts)
            eval, evec = la.eig(covar)
            m = np.mean(class_pts, axis=1)

            print('covar, eval-evec, m', covar, eval, evec, m)

            spatial_prop = {'e0_div_e1': eval[0]/eval[1], 'e1_div_e2': eval[1]/eval[2]}

            spatial_props.append(spatial_prop)

        return spatial_props


    def prepare_from_real_data(mat_file):

        print "Opening ", mat_file
        with open(mat_file, 'r') as f:
            meta = json.load(f)
        
        K = np.array(meta["intrinsic_matrix"]).reshape((3,3))

        fx = K[0,0]
        fy = K[1,1]
        px = K[0,2]
        py = K[1,2]
        zfar = 6.0
        znear = 0.25
        tnear = 0.5
        tfar = 1.5

        parameters = np.zeros((8, ), dtype=np.float32)
        parameters[0] = fx
        parameters[1] = fy
        parameters[2] = px
        parameters[3] = py
        parameters[4] = znear
        parameters[5] = zfar
        parameters[6] = tnear
        parameters[7] = tfar

        intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

        rgb = cv2.imread(os.path.join('/media/shawnle/Data0/YCB_Video_Dataset/SLM_datasets/Exhibition/DUCK', '{:06d}-color.png').format(1))
        height = rgb.shape[0]
        width = rgb.shape[1]

        return {'meta': meta,
                'parameters' : parameters,
                'factor_depth' : meta["factor_depth"] ,
                'intrinsic_matrix' : intrinsic_matrix,
                'class_indexes' : meta["cls_indexes"],
                'height' : height,
                'width' : width
            }

    cfg = config()
    imdb = get_imdb(cfg.imdb_name)
    
    from_real_mat = prepare_from_real_data(mat_file)
    from_json = prepare_from_json(imdb)

    points = imdb._points_all
    num_classes = imdb.num_classes -1

    # print('points max min:', np.amax(points), np.amin(points))
    # print('points shape:', points.shape)
    # print('points[1,...] :', points[1,...])

    spatial_props = calc_spatial_prop(points)
    
    return { 
            'num_classes' : num_classes,
            'points' : points,
            'from_json' : from_json,
            'from_meta' : from_real_mat,
            'num_cls' : points.shape[0]-1,
            'spatial_props' : spatial_props,
            'cfg' : cfg, 
            'imdb' : imdb
            }

def writePointsToFile(points, file_name):
    assert points.shape[0] == 3
    with open(file_name, 'w') as f:
        for i in range(points.shape[1]):
            f.write('{} {} {}\n'.format(points[0,i],points[1,i],points[2,i]))

def _load_object_points(xyz_file):
    '''
       points_all shape (num,3)
    '''

    print '[_load_object_points] starts'

    points = []
    num = np.inf

    point_file = xyz_file
    print 'point_file = ' + point_file
    assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
    points = np.loadtxt(point_file)
    if points.shape[0] < num:
        num = points.shape[0]

    points_all = np.zeros((num, 3), dtype=np.float32)
    points_all[:, :] = points[:num, :]

    return points_all


def blend_background(im_syn, im_bkgnd, backgrounds, syn_cfg):

    # sample a background image
    rgba = im_syn

    if syn_cfg["random_background"]:
        
        ind = np.random.randint(len(backgrounds), size=1)[0]
        filename = backgrounds[ind]
        print("background is %d named %s chosen" % (ind, filename))                        
    else:

        if not syn_cfg['syn_merges_real']:
            ind = syn_cfg["background_id"]
            filename = backgrounds[ind]
            print("background %d named %s is chosen" % (ind, filename))     

            background = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            try:
                background = cv2.resize(background, (rgba.shape[1], rgba.shape[0]), interpolation=cv2.INTER_LINEAR)
            except:
                background = np.zeros((rgba.shape[0], rgba.shape[1], 3), dtype=np.uint8)
                print 'bad background image'
        else:
            background = im_bkgnd
  
    # add background
    im = np.copy(rgba[:,:,:3])
    alpha = rgba[:,:,3]
    I = np.where(alpha == 0)
    # if cfg.INPUT == 'DEPTH' or cfg.INPUT == 'NORMAL':
    #     im_depth[I[0],I[1]] = background[I[0], I[1]] / 10
    # else:
    im[I[0], I[1], :] = background[I[0], I[1], :3]

    return im


class AugmentMerge_AcP_Syn():

    def __init__(self):

        self.process()

        return None


    def call_synthesizer(self, im_bkgnd, backgrounds):

        pr = self.augment_require_params
        width  =  pr['from_meta']['width']
        height =  pr['from_meta']['height']
        parameters = pr['from_meta']['parameters']
        print('width | height = ', width, height)
        num_classes = pr['num_cls']
        num_instances = pr['from_json']['num_instances']

        cad_name = 'data/LOV/models.txt'
        pose_name = 'data/LOV/poses.txt'
        synthesizer_ = libsynthesizer.Synthesizer(cad_name, pose_name)
        synthesizer_.setup(width, height)
        synthesizer_.init_rand(1200)

        # render a synthetic image
        im_syn = np.zeros((height, width, 4), dtype=np.float32)
        depth_syn = np.zeros((height, width, 3), dtype=np.float32)
        vertmap_syn = np.zeros((height, width, 3), dtype=np.float32)
        class_indexes = -1 * np.ones((num_classes, ), dtype=np.float32)
        poses = np.zeros((num_classes, 7), dtype=np.float32)
        centers = np.zeros((num_classes, 2), dtype=np.float32)
        is_sampling = True
        is_sampling_pose = True
        synthesizer_.render_python(int(width), int(height), parameters, num_instances, \
                                im_syn, depth_syn, vertmap_syn, class_indexes, poses, centers, is_sampling, is_sampling_pose)

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
        num = len(index)
        qt = np.zeros((3, 4, num), dtype=np.float32)

        im = blend_background(im_syn, im_bkgnd, backgrounds, pr['from_json']['syn_cfg'])

        return {'im_syn' : im#,
                # 'depth_syn' : depth_syn, 
                # 'vertmap_syn' : vertmap_syn,
                # 'class_indexes' : class_indexes,
                # 'poses' : poses,
                # 'centers' : centers
                }


    def process_selectModelPoints(self):
        '''
            selMdlPoints does not involve real data classes and instances since real data is already completed and no need further augmentation
                ** implication:
                    + num_classes = syn_num_classes + real_num_classes
        '''
        pr = self.augment_require_params
        kpt_num = pr['from_json']['num_kpt']
        num_classes = pr['num_cls']

        if pr['from_json']['syn_cfg']['selectModelPointsFromFile']:
            print('Function under QC')          

            selMdlPoints = np.zeros([3, num_classes * kpt_num], dtype=np.float32)
            for i in range(pr['num_cls']):
                file_name = osp.join('selected_points', 'model_{}_points.xyz'.format(i+1))

                selMdlPoints[:, i*kpt_num:(i+1)*kpt_num] = _load_object_points(file_name).T

        else:

            if pr['from_json']['syn_cfg']["sel_random_points"]:
                # selMdlPoints = selectModelPoints(pr['num_cls'], pr['from_json']['num_kpt'], pr['points'][1:,:,:])  # not use the background
                selMdlPoints = np.zeros([3, num_classes * kpt_num], dtype=np.float32)
                for i in range(pr['num_cls']):
                    selMdlPoints[:,i*kpt_num:(i+1)*kpt_num] = selectModelPointsBySpatialProp(pr['from_json']['num_kpt'], pr['points'][i+1,:,:], pr['spatial_props'][i])  # not use the background
                    print('selected mdl pts')
                    print(selMdlPoints[:,i*kpt_num:(i+1)*kpt_num])

                # export selected points to .xyz
                if  not os.path.exists('selected_points'):
                    os.makedirs('selected_points')
                else:
                    print('Please backup model point folder before continue.')
                    exit()

                for i in range(pr['num_cls']):
                    file_name = osp.join('selected_points', 'model_{}_points.xyz'.format(i+1))
                    writePointsToFile(selMdlPoints[:,i*kpt_num:(i+1)*kpt_num], file_name)

            else:
                selMdlPoints = pr['from_json']['syn_cfg']["selectModelPoints"]

                print (selMdlPoints)
                print (np.array(selMdlPoints).shape)
                shape = np.array(selMdlPoints).shape

                assert (shape[1] == 3) #, "current shape[1] = {}", (shape[1])
                assert (shape[0] == num_cls * num_kpt)

                selMdlPoints = np.array(selMdlPoints)
                selMdlPoints = np.transpose(selMdlPoints) #reshape((3, num_cls * num_kpt))

        return selMdlPoints


    def process_keypoint_projection(self):



        return None


    def process(self):
        '''
        read all images
        '''

        # DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/PoseCNN_Dataset"
        # DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data_syn_LOV/Loc_data"
        # DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data/0002/"
        # DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/SLM_datasets/Exhibition/DUCK"
        DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/SLM_datasets/Exhibition/POSE_iPBnet_20190621_water_bottle"
        # p = Path(DATA_ROOT)
        # p = osp.dirname(DATA_ROOT)

        # augment_require_mat = "{}/{:06d}-meta.mat".format(DATA_ROOT,1)
        augment_require_mat = "{}/{:06d}-meta.json".format(DATA_ROOT,1)
        augment_require_params = prepare_augment_from_real_data(augment_require_mat)
        self.augment_require_params = augment_require_params
        pr = augment_require_params

        # check output dir
        if not os.path.exists(pr['cfg'].root):
            os.makedirs(pr['cfg'].root)

        # load background
        # cache_file = pr['cfg'].BACKGROUND
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         backgrounds = cPickle.load(fid)
        #     print 'backgrounds loaded from {}'.format(cache_file)

        # select random points
        kpt_num = pr['from_json']['num_kpt']
        num_classes = pr['num_cls']

        assert not (pr['from_json']['syn_cfg']["sel_random_points"] and pr['from_json']['syn_cfg']['selectModelPointsFromFile']), 'Mode is not supported.'

        selMdlPoints = self.process_selectModelPoints()

        # iterate and project keypoints
        syn_cfg = pr['from_json']['syn_cfg']
        intrinsic_matrix = pr['from_meta']['intrinsic_matrix']
        height = pr['from_meta']['height']
        width = pr['from_meta']['width']
        class_indexes = pr['from_meta']['class_indexes']
        num_instances = pr['from_json']['num_instances']
        num_cls = pr['num_cls']
        num_kpt = pr['from_json']['num_kpt']
        factor_depth = pr['from_meta']['factor_depth']
        root = pr['cfg'].root
        imdb = pr['imdb']
        cfg = pr['cfg']

        # load background
        cache_file = cfg.BACKGROUND
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                backgrounds = cPickle.load(fid)
            print 'backgrounds loaded from {}'.format(cache_file)

        for i in range(cfg.data_num):  #cfg.data_num

            # read from real data -- start
            rgb_name = "{}/{:06d}-color.png".format(DATA_ROOT,i)
            im = cv2.imread(rgb_name, cv2.IMREAD_UNCHANGED)
            im_test = np.array(im, copy=True)

            # depth_name = "{}/{:06d}-depth.png".format(DATA_ROOT,i)
            # im_depth_raw = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED).astype(dtype=np.float32)   # BUG?!

            # print (p.joinpath("{:06d}-meta.mat".format(x)))
            mat_file = "{}/{:06d}-meta.json".format(DATA_ROOT,i)
            print "Opening ", mat_file
            with open(mat_file, 'r') as f:
                meta_dat = json.load(f)
            print (meta_dat)
            # read from real data -- end

            syn_num_cls = pr['num_classes']
            num = syn_num_cls
            # poses = np.array(meta_dat['poses']).reshape((4,4,num))
            # print('poses', poses)

            intrinsic_matrix = np.array(meta_dat['intrinsic_matrix']).reshape((3,3))

            # synthesize a frame -- start
            self.call_synthesizer(im, backgrounds)

            # synthesize a frame -- end
            exit()

            index = np.where(class_indexes >= 0)[0]
            print("class_indexes=", class_indexes)
            print("index=",index)

            sum_num_inst = np.sum(pr['from_json']['num_instances']).astype(int)
            qt = np.zeros((3, 4, sum_num_inst), dtype=np.float32)
            print('qt',qt)
            print('qt shape',qt.shape)
            print('sum_num_inst', sum_num_inst)

            # whole set of poses, each has whole bounding-box inside FOV
            set_is_qualified = True
            inst = 0
            for j in xrange(num):
                for k in xrange(num_instances[j]):
                    ind = index[j]
                    print('j, inst',j, inst)
                    qt[:, :3, inst] = poses[:3,:3,j] #quat2mat(poses[inst, :4])
                    qt[:, 3, inst] = poses[:3,3,j] #poses[inst, 4:]

                    inst = inst + 1

                    if syn_cfg["check_whole_bb"]:
                        if not is_qualified(qt[:,:,inst], extents[j,:], intrinsic_matrix, height, width):
                            set_is_qualified = False
                            break

            if not set_is_qualified:
                continue   

            # metadata
            # print('poses', qt.tolist())
            # print('cls_indexes', class_indexes)
            # print('cls_indexes', (class_indexes[index].astype(int) + 1).tolist())
            # metadata = {'poses': qt.tolist(), 'cls_indexes': (class_indexes[index].astype(int) + 1).tolist()}
            metadata = {'poses': qt.tolist(), 'cls_indexes': [class_indexes[id] + 1 for id in range(len(class_indexes)) ]}
            # print( 'metadata', metadata)

            # sampling and project model points
            Y2_meta = []

            inst = 0 
            for id in range(num_cls): # num_cls
                for jj in xrange(num_instances[id]): # num_instances[id]

                    inst_rec = []
                    RT = qt[:,:,inst]

                    print('project for class ' + str(id))
                    P3d = np.ones((4, num_kpt), dtype=np.float32)
                    P3d[:3,:] = selMdlPoints[:, id*num_kpt : (id+1)*num_kpt]
                    print('P3d ', P3d)

                    P3d_c = np.matmul(RT, P3d)
                    p2d = np.matmul(intrinsic_matrix, P3d_c)
                    p2d[0, :] = np.divide(p2d[0, :], p2d[2, :])
                    p2d[1, :] = np.divide(p2d[1, :], p2d[2, :])

                    print('p2d ', p2d)
                    print('K', intrinsic_matrix)
                    print('RT', RT)

                    for ii in range(p2d.shape[1]): # p2d.shape[1]

                        p = p2d[:2,ii]
                        status = checkVisibility(width, height, P3d_c[:,ii], p, im_depth_raw / factor_depth)
                        # Y2_meta.append( (p2d[0,ii], p2d[1,ii], status, id, inst) )
                        inst_rec.append( (p2d[0,ii], p2d[1,ii], status, id, inst) )

                        if status == 2:
                            print ('some point visible')
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

            # save meta_data
            filename = root + '{:06d}-meta_cpm.json'.format(i)
            with open(filename, 'w') as fp:
                json.dump(metadata, fp)
            print(metadata)


if __name__ == '__main__':

    augmenter = AugmentMerge_AcP_Syn()