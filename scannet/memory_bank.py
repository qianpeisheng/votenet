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
from model_util_scannet_CIL import ScannetDatasetConfig
DC = ScannetDatasetConfig()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Memory_bank():
    def __init__(self, total_budget, dataset) -> None:

        # total budget is the total number of objects in the memory bank
        self.total_budget = total_budget

        # TRAIN_NUM_OBJECTS_BY_CLASS is a list of the number of objects in each class, e.g., [100, 200, 300, 400, 500]
        self.TRAIN_NUM_OBJECTS_BY_CLASS = DC.train_num_obj_by_cls

        self.all_scan_names = dataset.all_scan_names
        self.scan_names = list(set(self.all_scan_names))
        self.all_base_classes = [] # initialization. This will be updated in incremental learning stages.

        # all objects
        self.pc_objects = [] # to be filled-in in the future

        print(f'An EMPTY memory bank is created with total budget: {self.total_budget}')

        self.object_memory_bank = [] # can be optimized using a multi indexed dictionary
        self.scene_memory_bank = []

        self.data_path = os.path.join(BASE_DIR, 'scannet_train_detection_data') # the memory bank is for the train set only.

        self.__fill_in_pc_objects__()

    def __fill_in_pc_objects__(self):
        # fill in the pc_objects list
        for index, scan_name in enumerate(self.scan_names):
            if index % 100 == 0:
                print('Memory bank initialization: {0}/{1} scenes'.format(index, len(self.scan_names)))
            instance_bboxes = self._get_bbox_list(scan_name, self.data_path)
            for idx, instance_bbox in enumerate(instance_bboxes):
                object_dict = {'scene_name': scan_name, 'object_id': idx, 'object_class': DC.nyu40id2class[instance_bbox[-1]]}
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

        # if the current budget is larger than the number of objects in the current_base_cls_pc_objects, set the current budget to the number of objects.
        current_budget = min(current_budget, len(current_base_cls_pc_objects))
        print(f'{current_budget} objects of classes {increment_classes} is added to the memory bank.')

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

        print(f'Current memory bank size is {len(self.object_memory_bank)}')
        # assert len(self.object_memory_bank) == self.total_budget, f'The number of objects in the memory bank {len(self.object_memory_bank)} is not equal to the total budget {self.total_budget}.'

    def get_current_budget(self):
        num_trues = 0
        for pc_object in self.pc_objects:
            if pc_object.is_in_memory_bank:
                num_trues += 1
        return num_trues

    def get_all_scan_names_with_True_values(self):
        # return all scene names that contain objects in the memory bank
        return self.scene_memory_bank

    def __getitem__(self, index):
        # TODO should we return the object or the scene?
        return self.pc_objects[index]

    def __len__(self):
        return len(self.pc_objects)
    # note this includes all objects, not just the objects in the memory bank.

    def save_memory_bank(self, save_path, stage_idx):
        # save the memory bank to 2 csv files, one for the object memory bank and one for the scene memory bank.
        object_memory_bank_file = os.path.join(save_path, f'stage_{stage_idx}_object_memory_bank.csv')
        scene_memory_bank_file = os.path.join(save_path, f'stage_{stage_idx}_scene_memory_bank.csv')

        with open(object_memory_bank_file, 'w') as f:
            f.write('scene_name,object_id,object_class\n')
            for pc_object in self.object_memory_bank:
                f.write(f'{pc_object.scene_name},{pc_object.object_id},{pc_object.object_class}\n')

        with open(scene_memory_bank_file, 'w') as f:
            for scene_name in self.scene_memory_bank:
                f.write(f'{scene_name}\n')

    def load_memory_bank(self, load_path):
        # load the memory bank from 2 csv files, one for the object memory bank and one for the scene memory bank.
        object_memory_bank_file = os.path.join(load_path, 'object_memory_bank.csv')
        scene_memory_bank_file = os.path.join(load_path, 'scene_memory_bank.csv')

        with open(object_memory_bank_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split(',')
                object_dict = {'scene_name': line[0], 'object_id': int(line[1]), 'object_class': int(line[2])}
                pc_object = PC_object(object_dict)
                self.pc_objects.append(pc_object)
                self.object_memory_bank.append(pc_object)

        with open(scene_memory_bank_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scene_memory_bank.append(line.strip())

if __name__=='__main__':

    # set_base_novel_scannet(num_base=9, num_novel=9)
    # memory_bank = Memory_bank(total_budget=100)
    # new_dict = set_true_values_in_sublists(dictionary=memory_bank.memory_bank, total_true= 200)
    import pdb; pdb.set_trace()