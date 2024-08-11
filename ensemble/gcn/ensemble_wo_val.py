import argparse
import pickle

import numpy as np
from tqdm import tqdm

import statistics

PATH_JOINT = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/joint/epoch_0_0.29526257803892764.pkl"
PATH_JOINT_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/joint_motion/epoch_0_0.08152772677194271.pkl"
PATH_BONE = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/bone/epoch_0_0.21263312522952627.pkl"
PATH_BONE_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/bone_motion/epoch_0_0.07895703268453912.pkl"

SAVE_PATH_FOLDER  = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/prediction"

label = open('/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SL-GCN/sign/27/val_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(PATH_JOINT, 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(PATH_JOINT_MOTION, 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(PATH_BONE, 'rb')
r3 = list(pickle.load(r3).items())
r4 = open(PATH_BONE_MOTION, 'rb')
r4 = list(pickle.load(r4).items())


def score_diff(score ,l):
    actuall = np.zeros(2000)
    actuall[l] = 1
    tmp = score + np.min(score)
    ret = abs((actuall - ((score-np.min(score))/(np.max(score)-np.min(score))))).sum()
    return ret


alpha = [1.0,0.9,0.5,0.5] # used in submission 1

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0
new_losses = []
with open(SAVE_PATH_FOLDER + '/predictions_wo_val.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        # print(name, name1, name2, name3, name4)
        assert name == name1 == name2 == name3 == name4
        mean += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
        # score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).mean()
        # score = r11*alpha[0] 
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(score)
        scores.append(score)
        preds.append(r)
        right_num += int(r == int(l))
        total_num += 1
        f.write('{}, {}\n'.format(name, r))
        new_losses.append(score_diff(score, int(l)))
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)
    print(f'Loss: {statistics.mean(new_losses)}')

f.close()
print(mean/len(label[0]))
# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

with open(SAVE_PATH_FOLDER + '/gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)