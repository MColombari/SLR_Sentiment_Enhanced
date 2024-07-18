import pickle
import csv

"""
Create the .pkl file for the testing phase
"""
TEST_LABEL_FILE_PATH = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/test_labels.csv'
DST_FILE_PATH = '../test_labels_WLASL.pkl'

file_names = []
print('start load test')
with open(TEST_LABEL_FILE_PATH, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        name, _ = row[0].split(',')
        
        file_names.append(name)

with open(DST_FILE_PATH, "wb") as f:
    pickle.dump(file_names, f)
print('finish process')
        