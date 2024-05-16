# coding: utf-8

""" This file implements a memory bank for the ScanNet dataset.
    It maintains:
        1. A list of all pc objects in the train set to choose from.
        2. A list of all scenes to choose from.
        3. A memory bank (list) that saves selected objects as replay examples.
        4. A memory bank (list) that saves scans containing the replayed objects.

    We use the name pc_objects to avoid confusion with the term object in Python.

    In this class, the class index are 0, 1, 2, ... not NYU40 class ids.
"""

import os
import sys
import numpy as np
import random
from PC_object import PC_object
from model_util_scannet_CIL_35 import ScannetDatasetConfig
import pickle
DC = ScannetDatasetConfig()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
random.seed(SEED)

def get_object_point_cloud(mesh_vertices, object_center, object_size):
    '''
    Get the point cloud of an object given the mesh vertices of the scene, the object center and the object size.
    '''
    # mesh_vertices is a numpy array of shape (n, 3), where n is the number of vertices in the mesh.
    # object_center is a numpy array of shape (3,) representing the center of the object.
    # object_size is a numpy array of shape (3,) representing the size of the object.
    # object_point_cloud is an numpy array of shape (n, 3), where n is the number of points within the bounding box.
    # object_point_cloud = np.array([point for point in mesh_vertices if np.abs(point[0] - object_center[0]) < object_size[0] / 2 and np.abs(point[1] - object_center[1]) < object_size[1] / 2 and np.abs(point[2] - object_center[2]) < object_size[2] / 2])
    object_point_cloud = np.array([point for point in mesh_vertices if np.abs(point[0] - object_center[0]) < object_size[0] / 2 and np.abs(point[1] - object_center[1]) < object_size[1] / 2 and np.abs(point[2] - object_center[2]) < object_size[2] / 2])
    return object_point_cloud

def create_and_save_object_reservoir(load_path, save_path):
    '''
    Loop through all scannet training files, get the object label, bounding box and the point cloud of the object.
    Save them to a file.
    '''

    all_scan_names = [f.split('_bbox')[0] for f in os.listdir(load_path) if f.endswith('_bbox.npy')]
    all_scan_names = [scan_name.split('_bbox')[0] for scan_name in all_scan_names]
    object_reservoir = []
    for scan_name in all_scan_names:
        print(f'Processing {scan_name}')
        instance_bboxes = np.load(os.path.join(load_path, scan_name) + '_bbox.npy')
        # an instance bbox is [center_x, center_y, center_z, length, width, height, class_id]
        mesh_vertices = np.load(os.path.join(load_path, scan_name) + '_vert.npy')
        for idx, instance_bbox in enumerate(instance_bboxes):
            print(f'Processing object {idx}')
            try:
                nyu_id = int(instance_bbox[-1])
                object_class = DC.nyu40id2class[nyu_id]
            except KeyError:
                # wall, ceiling and floor are excluded as they do not have instance labels. ---- They are not objects.
                # otherprop, other structure are excluded because they cover a wide range of objects. ---- They cannot be classified into a single class (very low mAP).
                continue
            object_dict = {'scene_name': scan_name, 'object_id': idx, 'object_class': object_class, 'nyu_id': nyu_id}
            object_center = instance_bbox[:3]
            object_size = instance_bbox[3:6]
            # objecct_point_cloud is an numpy array of shape (n, 3), where n is the number of points within the bounding box.
            object_point_cloud = get_object_point_cloud(mesh_vertices, object_center, object_size)
            object_dict['object_point_cloud'] = object_point_cloud
            object_reservoir.append(object_dict)

    # save the object_reservoir to a .pth file
    save_file_name = 'object_reservoir_35.pth'
    with open(os.path.join(save_path, save_file_name), 'wb') as f:
        pickle.dump(object_reservoir, f)
    print(f'Object reservoir is saved to {save_file_name}')

class Memory_Bank_Object():
    def __init__(self, total_budget, load_path) -> None:

        # total budget is the total number of objects in the memory bank
        self.total_budget = total_budget

        # load_path is the path to the object_reservoir_35.pth file
        self.load_path = load_path
        self.__load_object_reservoir__(load_path)

        # TRAIN_NUM_OBJECTS_BY_CLASS is a list of the number of objects in each class, e.g., [100, 200, 300, 400, 500]
        self.TRAIN_NUM_OBJECTS_BY_CLASS = DC.train_num_obj_by_cls

        # create a one_hot mask for the objects in the memory bank
        self.object_reservoir_length = len(self.object_reservoir)
        self.one_hot_mask = np.zeros(self.object_reservoir_length)

        self.classes = []

    def __load_object_reservoir__(self, load_path):
        with open(load_path, 'rb') as f:
            object_reservoir = pickle.load(f)
        self.object_reservoir = object_reservoir

    def __get_budget__random__(self, classes):
        '''
        Get the number of objects to be added to the memory bank for each class.
        Args:
            classes: a list of classes. Each class is the class of the objects to be added to the memory bank.
        '''
        new_class_all_class_ratio = len(classes) / len(self.classes)
        budget = int(self.total_budget * new_class_all_class_ratio)
        return budget

    def update_memory(self, budget, classes, criteria):
        '''
        Update the memory bank by adding or removing objects given the budget, the classes and the criteria.
        '''
        self.classes.extend(classes)
        if budget == -1:
            budget = self.__get_budget__random__(classes)
        print(f'Adding {budget} objects to the memory bank.')
        self.__remove_objects__(budget, classes, criteria)
        self.__add_objects__(budget, classes, criteria)

        # mask self.object_reservoir with self.one_hot_mask
        self.masked_object_reservoir = [obj for obj, mask in zip(self.object_reservoir, self.one_hot_mask) if mask == 1]

    def __add_objects__(self, budget, classes, criteria):
        '''
        Add objects to the memory bank given the budget, the classes and the criteria.
        Args:
            budget: a list of numbers. Each number is the number of objects to be added to the memory bank for each class.
            classes: a list of classes. Each class is the class of the objects to be added to the memory bank.
            criteria: a string. The criteria to add objects to the memory bank. Supporting 'random' only for now.
        The length of budget and classes should be the same.
        '''
        if criteria == 'random':
            if isinstance(budget, list):
                for _budget, _class in zip(budget, classes):
                    # get the indexes of the objects in the object_reservoir that belong to each classes.
                    # an element in the object_reservoir is a dictionary with keys 'scene_name', 'object_id', 'object_class' and 'object_point_cloud'.
                    class_indexes = [idx for idx, obj in enumerate(self.object_reservoir) if obj['object_class'] == _class
                                        and self.one_hot_mask[idx] == 0]
                    # randomly select _budget objects from the class_indexes.
                    selected_indexes = random.sample(class_indexes, _budget)
                    # set the one_hot_mask to 1 for the selected indexes.
                    self.one_hot_mask[selected_indexes] = 1
            elif isinstance(budget, int): # the budget is for all classes
                # get the indexes of the objects in the object_reservoir that belong to any of the classes.
                class_indexes = [idx for idx, obj in enumerate(self.object_reservoir) if obj['object_class'] in classes
                                    and self.one_hot_mask[idx] == 0]
                # randomly select budget objects from the class_indexes.
                selected_indexes = random.sample(class_indexes, budget)
                # set the one_hot_mask to 1 for the selected indexes.
                self.one_hot_mask[selected_indexes] = 1
            else:
                raise ValueError('Undefined type of budget.')
        else:
            raise ValueError('Undefined criteria for adding objects.')

    def __remove_objects__(self, budget, classes, criteria):
        '''
        Remove objects from the memory bank given the budget, the classes and the criteria.
        Args:
            budget: the number of objects to be added to the memory bank.
            classes: the classes of the objects to be added to the memory bank. Unused for now.
            criteria: the criteria to remove objects from the memory bank. Supporting 'random' only for now.
        '''
        # calculate the size of the memory bank currently, which is the number of 1s in the one_hot_mask.
        current_memory_bank_size = np.sum(self.one_hot_mask)
        # if the number of objects in the memory bank is less than or equal to self.total_budget - budget, return.
        if current_memory_bank_size <= self.total_budget - budget:
            return True
        else:
            # calculate how many objects need to be removed.
            num_objects_to_remove = int(current_memory_bank_size - (self.total_budget - budget))
            # remove objects in the memory bank according to the criteria.
            # so that the number of objects in the memory bank is equal to self.total_budget - budget.
            if criteria == 'random':
                # randomly remove num_objects_to_remove objects from the memory bank.
                indices = np.where(self.one_hot_mask == 1)[0]
                indices_to_remove = random.sample(list(indices), num_objects_to_remove)
                self.one_hot_mask[indices_to_remove] = 0
            else:
                # throw an error if the criteria is not 'random'.
                raise ValueError('Undefined criteria for removing objects.')

            return False


    def __getitem__(self, index):
        # return the inndex of the index-th 1 in the one_hot_mask.
        return self.masked_object_reservoir[index]

    def __len__(self):
        return int(np.sum(self.one_hot_mask)) # this is the same as the length of self.masked_object_reservoir
    # note this includes all objects, not just the objects in the memory bank.

    def save_memory_bank(self, save_path, stage_idx):
        '''
        Save one_hot_mask to npy file.
        '''
        np.save(os.path.join(save_path, f'one_hot_mask_{stage_idx}.npy'), self.one_hot_mask)

    def load_memory_bank(self, load_path):
        '''
        Load one_hot_mask from npy file.
        '''
        return np.load(load_path)

if __name__=='__main__':
    # pass

    # import pdb; pdb.set_trace()
    create_and_save_object_reservoir('scannet_train_detection_data_40', '.')
    # memory_bank = Memory_Bank_Object(100, '.')