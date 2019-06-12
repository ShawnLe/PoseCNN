'''
@description to read old exported PCNN real(not synthesized) data and export to train iPbNet workflow
'''
import os, sys
import _init_paths

from fcn.config import cfg, cfg_from_file, get_output_dir
import argparse

# from pathlib import Path
import scipy.io

from datasets.factory import get_imdb

'''
    read the model
'''
def _load_object_points(self):

    print ('[_load_object_points] starts')

    points = [[] for _ in xrange(len(self._classes))]
    num = np.inf

    for i in xrange(1, len(self._classes)):
        point_file = os.path.join(self._lov_path, 'models', self._classes[i], 'points.xyz')
        print ('point_file = ' + point_file)
        assert os.path.exists(point_file), 'Path does not exist: {}'.format(point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    points_all = np.zeros((self.num_classes, num, 3), dtype=np.float32)
    for i in xrange(1, len(self._classes)):
        points_all[i, :, :] = points[i][:num, :]

    return points, points_all


if __name__ == '__main__':

    '''
    read all images
    '''
    data_num = 25

    # DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/PoseCNN_Dataset"
    DATA_ROOT = "/media/shawnle/Data0/YCB_Video_Dataset/YCB_Video_Dataset/data_syn_LOV/Loc_data"
    p = Path(DATA_ROOT)

    meta_dat = []
    for x in range(1): # 347
        print (p.joinpath("{:06d}-meta.mat".format(x)))
        # meta_dat = scipy.io.loadmat(p.joinpath("{:06d}-meta.mat".format(x)))
        meta_dat.append(scipy.io.loadmat("{}/{:06d}-meta.mat".format(DATA_ROOT,x)))
        # print (meta_dat)

    print(meta_dat[0]['cls_indexes'])
    print(meta_dat[0]['poses'])

    # iterate and project keypoints



    # export meta files