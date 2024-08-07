import scipy
import csv
import numpy as np

"""
Create the .mat file for the training phase
"""

TRAIN_LABEL_FILE_PATH = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/train_labels.csv'
TEST_LABEL_FILE_PATH = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/test_labels.csv'

REMOVED_FILES = ['signer11_sample59', 'signer31_sample81','signer5_sample1557']

out_dict = {}
out_dict['test_count'] = 0
out_dict['test_file_name'] = []
out_dict['test_label'] = []
out_dict['train_count'] = 0
out_dict['train_file_name'] = []
out_dict['train_label'] = []

print('start train')
with open(TRAIN_LABEL_FILE_PATH, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        name, label = row[0].split(',')

        if name in REMOVED_FILES:
            continue

        out_dict['train_count'] += 1
        #out_dict['train_file_name'].append([np.array([[np.array([name], dtype='<U18')]], dtype=object)])
        out_dict['train_file_name'].append(name)
        out_dict['train_label'].append(int(label))

print('start val')
with open(TEST_LABEL_FILE_PATH, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        name, label = row[0].split(',')

        if name in REMOVED_FILES:
            continue

        out_dict['test_count'] += 1
        out_dict['test_file_name'].append(name)
        out_dict['test_label'].append(int(label))

scipy.io.savemat('./train_val_split_WLASL_no_val.mat', mdict=out_dict)
print('done')
