import argparse
import pickle

import numpy as np
from tqdm import tqdm

#TODO change the paths with correct retrival paths
label = open('/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SL-GCN/sign/27/train_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open('joint_train.pkl', 'rb')
r1 = list(pickle.load(r1).items())
r2 = open('bone_train.pkl', 'rb')
r2 = list(pickle.load(r2).items())
r3 = open('joint_motion_train.pkl', 'rb')
r3 = list(pickle.load(r3).items())
r4 = open('bone_motion_train.pkl', 'rb')
r4 = list(pickle.load(r4).items())


alpha = [1.2,1.2,0.5,0.5] # used in submissions 3


names = []
scores = []
print(len(label[0]))

for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        assert name == name1 == name2 == name3 == name4
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum() 
        print('score.shape: ',score.shape)
        scores.append(score)

# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

with open("/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/Training_data/train_dataset/retrival_ensembled.pkl", 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)