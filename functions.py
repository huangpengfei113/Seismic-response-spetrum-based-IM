import numpy as np
import csv
import keras
from keras import Sequential
from keras import models, layers
from keras import optimizers
from keras.models import Input
from keras import regularizers
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf
import os


def read_csv_data(path_csv):
    with open(path_csv, 'r') as f:
        reader = csv.reader(f)
        data_in = [i for i in reader]
    return data_in


def read_file(file_dir):
    for root, dirs, files in os.walk(file_dir):
        return root, dirs, files


# transform List type to numpy.array type
def array_change(list_data):
    array_data = np.zeros(len(list_data), )
    for i, data in enumerate(list_data):
        array_data[i] = data
    array_data = array_data.reshape(len(list_data), )
    return array_data


# transform List type to numpy.array type
def array_change2(list_data):
    array_data = np.zeros((len(list_data), len(list_data[0])))
    for i, data in enumerate(list_data):
        for j, data_j in enumerate(data):
            array_data[i, j] = data_j
    return array_data


# shuffle array at the first dim
def shuffle_numpy(num_array1, num_array2, num_array3, num_array4, num_array5, num_array6, num_name6):
    array_shape = num_array1.shape
    array_len = array_shape[0]
    l_random = creat_shuffle(array_len)
    new_1 = np.zeros(num_array1.shape)
    new_2 = np.zeros(num_array2.shape)
    new_3 = np.zeros(num_array3.shape)
    new_4 = np.zeros(num_array4.shape)
    new_5 = np.zeros(num_array5.shape)
    new_6 = np.zeros(num_array6.shape)
    new_7 = ['0' for i in range(len(num_name6))]
    for count, i in enumerate(l_random):
        new_1[count] = num_array1[i]
        new_2[count] = num_array2[i]
        new_3[count] = num_array3[i]
        new_4[count][:] = num_array4[i][:]
        new_5[count][:] = num_array5[i][:]
        new_6[count][:] = num_array6[i][:]
        new_7[count] = num_name6[i]
    return new_1, new_2, new_3, new_4, new_5, new_6, new_7


def creat_shuffle(len1):
    origin_seq = [i for i in range(len1)]
    new_seq = []
    for i, j in enumerate(range(len1)):
        ran_num = np.random.randint(0, len1-i)
        new_seq.append(origin_seq[ran_num])
        origin_seq.remove(new_seq[-1])
    return new_seq


def return_log(y_pred):
    return K.log(K.clip(y_pred, K.epsilon(), None))
    

def gene_mlp(data_input_all, data_output_all, min_index, max_index, shuffle=False, start_index=0, batch_size=128):
    max_index_limit = len(data_input_all)
    if max_index is None:
        max_index = max_index_limit
    elif max_index > max_index_limit:
        print('The defined value is out of range, reset it to {}'.format(str(max_index_limit)))
        max_index = max_index_limit
    if min_index is None:
        min_index = 0
    i = start_index
    while 1:
        if_break = False
        if shuffle:
            wave_index = np.random.randint(min_index, max_index-1, size=batch_size)
        else:
            temp_index_max = len(data_input_all) - batch_size
            if i > temp_index_max:
                print("Out of range, the sample number is decreased")
                batch_size_new = batch_size - (i - temp_index_max)
                if_break = True
            if if_break:
                wave_index = [mm + i for mm in range(batch_size_new)]
                i = 0
            else:
                wave_index = [mm + i for mm in range(batch_size)]
                i += len(wave_index)

        if if_break:
            samples3 = np.zeros((batch_size_new, len(data_input_all[0])))
            targets = np.ones((batch_size_new,))
        else:
            samples3 = np.zeros((batch_size, len(data_input_all[0])))
            targets = np.ones((batch_size,))

        for j, row in enumerate(wave_index):
            input_temp = np.array(data_input_all[row])
            targets[j] = data_output_all[row]
            for mm in range(len(input_temp)):
                samples3[j, mm] = input_temp[mm]
        yield samples3, targets


def gene_mlp_judge(data_input_all, data_output_all, min_index, max_index, shuffle=False, start_index=0, batch_size=128):
    max_index_limit = len(data_input_all)
    if max_index is None:
        max_index = max_index_limit
    elif max_index > max_index_limit:
        print('The defined value is out of range, reset it to {}'.format(str(max_index_limit)))
        max_index = max_index_limit
    if min_index is None:
        min_index = 0
    i = start_index
    while 1:
        if_break = False
        if shuffle:
            wave_index = np.random.randint(min_index, max_index - 1, size=batch_size)
        else:
            temp_index_max = len(data_input_all) - batch_size
            if i > temp_index_max:
                print("Out of range, the sample number is decreased")
                batch_size_new = batch_size - (i - temp_index_max)
                if_break = True
            if if_break:
                wave_index = [mm + i for mm in range(batch_size_new)]
                i = 0
            else:
                wave_index = [mm + i for mm in range(batch_size)]
                i += len(wave_index)

        if if_break:
            samples3 = np.zeros((batch_size_new, len(data_input_all[0])))
            targets = np.ones((batch_size_new,))
        else:
            samples3 = np.zeros((batch_size, len(data_input_all[0])))
            targets = np.ones((batch_size,))

        for j, row in enumerate(wave_index):
            input_temp = np.array(data_input_all[row])
            targets[j] = data_output_all[row]
            for mm in range(len(input_temp)):
                samples3[j, mm] = input_temp[mm]
        yield samples3, targets, if_break
        

def write_csv(array1, path1):
    with open(path1, 'w', newline='') as f:
        writer1 = csv.writer(f)
        writer1.writerows(array1)


# get the maximum absolute value of a array
def get_max_abs(array1):
    abs_array = np.abs(array1)
    max_value = np.max(abs_array)
    return max_value


# transform str type to float type
def str_float(list1):
    for count1, i in enumerate(list1):
        for count2, j in enumerate(i):
            list1[count1][count2] = float(j)
    return list1

