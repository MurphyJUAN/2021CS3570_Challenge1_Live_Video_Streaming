import os
import random
import numpy as np
COOKED_TRACE_FOLDER = './cooked_traces/'


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER, shuffle=False):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []
        # print file_path
        with open(file_path, 'rb') as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)
    if shuffle:
        # Change to np
        all_cooked_time = np.array(all_cooked_time)
        all_cooked_bw = np.array(all_cooked_bw)
        all_file_names = np.array(all_file_names)
        #shuffle
        indices = np.arange(all_cooked_time.shape[0])
        np.random.shuffle(indices)
        all_cooked_time = all_cooked_time[indices].tolist()
        all_cooked_bw = all_cooked_bw[indices].tolist()
        all_file_names = all_file_names[indices].tolist()
        #change to list
    return all_cooked_time, all_cooked_bw, all_file_names
