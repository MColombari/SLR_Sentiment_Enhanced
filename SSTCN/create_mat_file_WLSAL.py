import scipy
import csv


TRAIN_LABEL_FILE_PATH = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/train_labels.csv'
VAL_LABEL_FILE_PATH = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/val_labels.csv'

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
        out_dict['train_count'] += 1
        out_dict['train_file_name'].append(name)
        out_dict['train_label'].append(label)

print('start val')
with open(VAL_LABEL_FILE_PATH, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        name, label = row[0].split(',')
        out_dict['test_count'] += 1
        out_dict['test_file_name'].append(name)
        out_dict['test_label'].append(label)


scipy.io.savemat('/homes/mcolombari/SLR_Sentiment_Enhanced/SSTCN/train_val_split_WLASL.mat', mdict=out_dict)
print('done')