# import libraries
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd
import os
import re

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization

import random
from timeit import default_timer as timer

import json

def GetID(Folder,data_txt_file):
    
    ID_file_path = ""

    #if type_of_data == 'train_no_threat':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\train_no_threat.txt'
    #elif type_of_data == 'train_threat':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\train_threat.txt'
    #elif type_of_data == 'train_threat_augment':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\train_threat_augment.txt'
    #elif type_of_data == 'dev_no_threat':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\dev_no_threat.txt'
    #elif type_of_data == 'dev_threat':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\dev_threat.txt'
    #elif type_of_data == 'test_no_threat':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\test_no_threat.txt'
    #elif type_of_data == 'test_threat':
    #    ID_file_path=r'I:\Krongrath\Transfer\dataset\test_threat.txt'
    ID_file_path = Folder+"/"+data_txt_file

    with open(ID_file_path, 'r') as f:
         IDlist = json.loads(f.read())

    return IDlist

#----------------------------------------------------------------------------
Folder= r'I:\Krongrath\Transfer\dataset'
train_threat = 'train_threat.txt'
train_threat_augment ='train_threat_augment.txt'
ID_list1 = GetID(Folder,train_threat)
ID_list2 = GetID(Folder,train_threat_augment)

#----------------------------------------------------------------------------
#get name of each ID 
def GetName(ID,image_angle):
    Name = 'ID_{}_angle_{}.jpg'.format(ID,image_angle)
    return Name

#----------------------------------------------------------------------------
# create the list of all IDs for 1 ID list (train_no_threat)
def create_ID_list(ID_list,type_of_image):
    all_ID = []
    #type_of_image = ['original','translate_y_up','translate_y_down','sharp','lightness','translate_sharp']
    for type in type_of_image:
        for ID in ID_list:
             single_ID = '{}_ID_{}'.format(type,ID)
             all_ID.append(single_ID)

    return all_ID

#----------------------------------------------------------------------------
#create the list of all IDs for 2 ID lists (train_threat)
def create_ID_list2(ID_list1,type_of_image_1,ID_list2,type_of_image_2):
    all_ID = []

    for ID in ID_list1:
        single_ID = '{}_ID_{}'.format(type_of_image_1,ID)
        all_ID.append(single_ID)

    for ID2 in ID_list2:
        single_ID2 = '{}_ID_{}'.format(type_of_image_2,ID2)
        all_ID.append(single_ID2)

    return all_ID

def create_ID_list_no_aug(ID_list):
    all_ID = []
    for ID in ID_list:
        single_ID = 'ID_{}'.format(ID)
        all_ID.append(single_ID)

    return all_ID

#all_ID = create_ID_list2(ID_list1,'original',ID_list2,'translate_y_up_down')
#print(all_ID)
