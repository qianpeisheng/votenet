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
from .PC_object import PC_object

class Memory_bank():
    def __init__(self, total_budget, dataset) -> None:
        
        # total budget is the total number of objects in the memory bank
        self.total_budget = total_budget

        # TRAIN_NUM_OBJECTS_BY_CLASS is a list of the number of objects in each class, e.g., [100, 200, 300, 400, 500]
        self.TRAIN_NUM_OBJECTS_BY_CLASS = dataset.dataset_config.TRAIN_NUM_OBJECTS_BY_CLASS

        # dataset_config_types is a list of class names, e.g., ['chair', 'table', 'sofa', 'bed', 'desk']
        self.dataset_config_types = dataset.dataset_config.types

        self.nyu40id2class = dataset.dataset_config.nyu40id2class

        # data_path_train is the path to the training data, used to create all objects 
        self.data_path_train = os.path.join(dataset.dataset_config.ROOT_DIR, 'scannet', 'scannet_train_detection_data')
        self.class2scans = dataset.class2scans
        self.all_scan_names = dataset.all_scan_names
        self.scan_names = list(set(self.all_scan_names))
        # self.current_classes = dataset.current_class_index_list
        self.all_base_classes = [] #dataset.current_class_index_list # initialization. This will be updated in incremental learning stages.
        
        # all objects
        self.pc_objects = [] # to be filled-in in the future
        
        # all scenes
        self.scenes = [] # The list contains all scenes from objects that is_in_memory_bank is True.

        print(f'An EMPTY memory bank is created with total budget: {self.total_budget}')

        self.object_memory_bank = []
        self.scene_memory_bank = []

    def __fill_in_pc_objects__(self):
        # fill in the pc_objects list
        for index, scan_name in enumerate(self.scan_names):
            if index % 100 == 0:
                print('Memory bank initialization: {0}/{1} scenes'.format(index, len(self.scan_names)))
            instance_bboxes = self._get_bbox_list(scan_name, self.data_path_train)
            for idx, instance_bbox in enumerate(instance_bboxes):
                object_dict = {'scene_name': scan_name, 'object_id': idx, 'object_class': self.nyu40id2class[instance_bbox[-1]]}
                pc_object = PC_object(object_dict)
                self.pc_objects.append(pc_object)

    def _get_bbox_list(self, scan_name, scan_data_path):
        instance_bboxes = np.load(os.path.join(scan_data_path, scan_name) + '_bbox.npy')
        return instance_bboxes
    
    def set_random_prob(self):
        # need to set in every stage because the total number of classes and objects changes.
        total_num_objects = sum([self.TRAIN_NUM_OBJECTS_BY_CLASS[i] for i in self.all_base_classes])
        self.random_prob = self.total_budget / total_num_objects

    def add_pc_object_to_memory_bank(self, pc_object):
        # add the pc_object to the memory bank
        pc_object.add_to_memory_bank()
        self.object_memory_bank.append(pc_object)

    def remove_pc_object_from_memory_bank(self, pc_object):
        # remove the pc_object from the memory bank
        pc_object.remove_from_memory_bank()

    def update_memory_bank_for_pc_objects(self):
        self.object_memory_bank = [pc_object for pc_object in self.pc_objects if pc_object.is_in_memory_bank]

    def update_memory_bank_scene_by_pc_objects(self):
        # update the scene memory bank based on the pc_objects
        # We do not update the scene memory bank when adding or removing objects from the object memory bank because
        # that would be less efficient.
        # Instead, we update the scene memory bank only when needed.
        # Note that this function redefines the scene memory bank instead of updating it.
        self.scene_memory_bank = list(set([pc_object.scene_name for pc_object in self.object_memory_bank]))

    def randomly_popolate(self, increment_classes):
        # This is the key step to generate the memory bank randomly.
        # First, get a sublist of the pc_objects, inlucde only the objects in the increment_classes.
        # Next, shuffle the list and select the first n objects, where n = self.total_budget * len(increment_classes) / len(self.all_base_classes)
        # Finally, remove n objects from the object_memory_bank and add the n objects to the object_memory_bank.
        
        # update self.all_base_classes
        self.all_base_classes += increment_classes
        current_base_cls_pc_objects = [pc_object for pc_object in self.pc_objects if pc_object.object_class in increment_classes]
        
        random.shuffle(current_base_cls_pc_objects)
        # print(increment_classes, self.all_base_classes)
        current_budget = int(self.total_budget * len(increment_classes) / len(self.all_base_classes))
        print(f'Randomly populate the memory bank with {current_budget} objects.')
        
        # if the current budget is larger than the number of objects in the current_base_cls_pc_objects, set the current budget to the number of objects.
        current_budget = min(current_budget, len(current_base_cls_pc_objects))
        
        if len(self.object_memory_bank) > 0:
            # shuffle self.object_memory_bank and select the first current_budget objects to delete
            random.shuffle(self.object_memory_bank)
            while len(self.object_memory_bank) > self.total_budget - current_budget:
                self.remove_pc_object_from_memory_bank(self.object_memory_bank[0])
                self.object_memory_bank.pop(0)
        for i in range(current_budget):
            self.add_pc_object_to_memory_bank(current_base_cls_pc_objects[i])
        # update the object memory bank list
        self.update_memory_bank_for_pc_objects()
        self.update_memory_bank_scene_by_pc_objects()

        assert len(self.object_memory_bank) == self.total_budget, f'The number of objects in the memory bank {len(self.object_memory_bank)} is not equal to the total budget {self.total_budget}.'

    def get_current_budget(self):
        # for all values in the dictionary, sum the number of True values in the third list
        num_trues = 0
        for key in self.memory_bank:
            num_trues += sum(self.memory_bank[key][2])
        return num_trues
    
    def get_total_number_of_objects(self):
        # for all values in the dictionary, sum the number of True values in the third list
        num_objects = 0
        for key in self.memory_bank:
            num_objects += len(self.memory_bank[key][2])
        return num_objects
    
    def get_memory_bank(self):
        return self.memory_bank
    
    def get_all_scan_names_with_True_values(self):
        # return all scene names that contain objects in the memory bank
        return self.scene_memory_bank
    
    def __getitem__(self, index):
        # TODO should we return the object or the scene?
        return self.pc_objects[index]

if __name__=='__main__':

    # set_base_novel_scannet(num_base=9, num_novel=9)
    # memory_bank = Memory_bank(total_budget=100)
    # new_dict = set_true_values_in_sublists(dictionary=memory_bank.memory_bank, total_true= 200)
    import pdb; pdb.set_trace()