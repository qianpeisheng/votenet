# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from box_util import get_3d_box

class ScannetDatasetConfig(object):
    def __init__(self):
        self.num_class = 35
        self.num_heading_bin = 1
        self.num_size_cluster = 35

        # CIL related settings
        self.CIL_stages=[[0,1,2,3,4,5,6],[7,8,9,10,11,12,13],[14,15,16,17,18,19,20],
                        [21,22,23,24,25,26,27],[28,29,30,31,32,33,34]] #7-7-7-7-7

        self.CIL_stages_str = self.get_CIL_stages_str()

        self.train_num_obj_by_cls = [4357, 2026, 1985, 1554, 1427, 1271, 928, 745, 661, 657, 551, 486, 481, 406, 390, 386, 364, 307, 300, 292, 279, 253, 251, 216, 201, 190, 186, 177, 170, 116, 113, 52, 39, 32, 22]

        self.type2class = {'chair':0, 'door':1, 'other furniture':2, 'books':3,
                            'cabinet':4, 'table':5, 'window':6, 'pillow':7,
                            'picture':8, 'box':9, 'desk':10, 'shelves':11, 'towel':12, 'sofa':13,
                            'sink':14, 'clothes':15, 'lamp':16, 'bed':17, 'bookshelf':18, 'curtain':19, 'mirror':20,
                            'bag':21, 'whiteboard':22, 'counter':23, 'toilet':24, 'nightstand':25, 'refrigerator':26,
                            'television':27, 'dresser':28, 'shower curtain':29, 'bathtub':30, 'paper':31, 'person':32,
                            'floor mat':33, 'blinds':34}
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.nyu40ids = np.array([ 5, 8, 39, 23, 3, 7,  9, 18, 11, 29, 14, 15, 27, 6, 34, 21, 35, 4, 10, 16, 19, 37, 30, 12, 33, 32, 24, 25, 17, 28, 36, 26, 31, 20, 13])
        self.nyu40id2class = {nyu40id: i for i,nyu40id in enumerate(list(self.nyu40ids))}
        self.mean_size_arr = np.load(os.path.join(ROOT_DIR,'scannet/meta_data/mean35.npz'))['arr_0']
        self.type_mean_size = {}
        for i in range(self.num_size_cluster):
            self.type_mean_size[self.class2type[i]] = self.mean_size_arr[i,:]

    # CIL methods
    def get_CIL_stages_str(self):
        # convert the nested list to a string
        stage_strings = []
        for stage_idx, stage in enumerate(self.CIL_stages):
            stage_str = f'Stage {stage_idx}: '
            for cls in stage:
                stage_str += str(cls) + ', '
            stage_strings.append(stage_str)
        return stage_strings

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class
            [optinal] also small regression number from
            class center angle to current angle.

            angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
            return is class of int32 of 0,1,...,N-1 and a number such that
                class*(2pi/N) + number = angle

            NOT USED.
        '''
        assert(False)

    def class2angle(self, pred_cls, residual, to_label_format=True):
        ''' Inverse function to angle2class.

        As ScanNet only has axis-alined boxes so angles are always 0. '''
        return 0

    def size2class(self, size, type_name):
        ''' Convert 3D box size (l,w,h) to size class and size residual '''
        size_class = self.type2class[type_name]
        size_residual = size - self.type_mean_size[type_name]
        return size_class, size_residual

    def class2size(self, pred_cls, residual):
        ''' Inverse function to size2class '''
        return self.mean_size_arr[pred_cls, :] + residual

    def param2obb(self, center, heading_class, heading_residual, size_class, size_residual):
        heading_angle = self.class2angle(heading_class, heading_residual)
        box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle*-1
        return obb

def rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))

    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]


    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)
