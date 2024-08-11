import argparse
import pickle

import numpy as np
from tqdm import tqdm
import csv
import json

import statistics


PATH_JOINT = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/joint/epoch_0_0.29526257803892764.pkl"
PATH_JOINT_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/joint_motion/epoch_0_0.08152772677194271.pkl"
PATH_BONE = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/bone/epoch_0_0.21263312522952627.pkl"
PATH_BONE_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/bone_motion/epoch_0_0.07895703268453912.pkl"
PATH_EMOTIONS = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/val.csv"
EMOTION_ASSOSIATION = "/homes/mcolombari/SLR_Sentiment_Enhanced/DAN_module/sign_emotion_assosiation.json"

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


emotions = dict()
with open(PATH_EMOTIONS, newline='\n') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        emotions[row[0]] = [
            int(row[1]),
            [float(row[2]),
             float(row[3]),
             float(row[4]),
             float(row[5]),
             float(row[6]),
             float(row[7]),
             float(row[8]),
             float(row[9])]]

f_w_to_e = open(EMOTION_ASSOSIATION)
w_to_e = json.load(f_w_to_e)


def enhance_emotion(score, emotion, label):
    e_weight = np.zeros(2000)
    for i in range(1,8):
        e_weight[w_to_e[str(i)]] = emotion[1][i]

    score += score * e_weight
    # print("after")
    # print(score[list_enhance])
    # print(score[list_reduce])
    return score


def score_diff(score ,l):
    actuall = np.zeros(2000)
    actuall[l] = 1
    tmp = score + np.min(score)
    ret = abs((actuall - ((score-np.min(score))/(np.max(score)-np.min(score))))).sum()
    return ret


alpha = [1.0,0.9,0.5,0.5] # used in submission 1

right_num_e = total_num_e = right_num_5_e = 0
names = []
preds_e = []
scores_e = []
mean_e = 0
new_losses = []
with open(SAVE_PATH_FOLDER + '/predictions_wo_val.csv', 'w') as f:
    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        # if name != "signer5_sample336":
        #     continue
        print(name)
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        # print(name, name1, name2, name3, name4)
        assert name == name1 == name2 == name3 == name4
        mean_e += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
        # print(len(score))

        # Motion enhance
        score_emotion = enhance_emotion(score, emotions[name], l)
        rank_5_e = score_emotion.argsort()[-5:]
        print(rank_5_e)
        right_num_5_e += int(int(l) in rank_5_e)
        r_e = np.argmax(score_emotion)
        scores_e.append(score_emotion)
        preds_e.append(r_e)
        right_num_e += int(r_e == int(l))
        total_num_e += 1
        f.write('{}, {}\n'.format(name, r_e))

        new_score = score_diff(score, int(l))
        print(new_score)
        new_losses.append(new_score)


    acc_e = right_num_e / total_num_e
    acc5_e = right_num_5_e / total_num_e
    print(total_num_e)
    print('top1: ', acc_e)
    print('top5: ', acc5_e)
    print(f'Loss: {statistics.mean(new_losses)}')

f.close()
f_w_to_e.close()
print(mean_e/len(label[0]))
# with open('./val_pred.pkl', 'wb') as f:
#     # score_dict = dict(zip(names, preds))
#     score_dict = (names, preds)
#     pickle.dump(score_dict, f)

with open(SAVE_PATH_FOLDER + '/gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores_e))
    pickle.dump(score_dict, f)