import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm
import csv
import json

import statistics


EXPERIMENT_NUMBER = 4

PATH_JOINT = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/joint/epoch_0_0.29526257803892764.pkl"
PATH_JOINT_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/joint_motion/epoch_0_0.08152772677194271.pkl"
PATH_BONE = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/bone/epoch_0_0.21263312522952627.pkl"
PATH_BONE_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/bone_motion/epoch_0_0.07895703268453912.pkl"
PATH_EMOTIONS = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/val.csv"
EMOTION_ASSOSIATION = "/homes/mcolombari/SLR_Sentiment_Enhanced/DAN_module/sign_emotion_assosiation.json"

EMOTION_SLOPE = 1.2
EMOTION_OFFSET = 0

SAVE_PATH_FOLDER  = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/prediction/" + str(EXPERIMENT_NUMBER)
if not os.path.exists(SAVE_PATH_FOLDER):
    os.makedirs(SAVE_PATH_FOLDER)

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

# print(w_to_e.values())

emotion_sensitive_label = []
for i in w_to_e.values():
    for a in i:
        emotion_sensitive_label.append(a)

# print(emotion_sensitive_label)


def enhance_emotion(score, emotion, label):
    top10 = score.argsort()[-10:]
    e_weight = np.zeros(2000)
    for i in range(1,8):
        possible_label = w_to_e[str(i)]
        use_lable = []
        for e in top10:
            if e in possible_label:
                use_lable.append(e)
        
        e_weight[use_lable] = emotion[1][i]

    ret = score.copy()
    ret += ret * ((e_weight * EMOTION_SLOPE) + EMOTION_OFFSET)
    # print("after")
    # print(score[list_enhance])
    # print(score[list_reduce])
    return ret


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def score_diff(score ,l):
    actuall = np.zeros(2000)
    actuall[l] = 1
    tmp = score + np.min(score)
    ret = abs((actuall - ((score-np.min(score))/(np.max(score)-np.min(score))))).sum()
    return ret


def score_diff_with_softmax(score ,l):
    actuall = np.zeros(2000)
    actuall[l] = 1
    tmp = score + np.min(score)
    score_with_softmax = softmax(score)
    # print(norm)
    ret = abs((actuall - score_with_softmax)).sum()
    return ret


alpha = [1.0,0.5,1.0,0.5]

right_num_e = total_num_e = right_num_5_e = 0
names = []
preds_e = []
scores_e = []
mean_e = 0
new_losses = []
new_losses_softmax = []
with open(SAVE_PATH_FOLDER + '/predictions_wo_val.csv', 'w') as f:
    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        # if name != "signer5_sample336":
        #     continue
        # print(name)

        # Only for emotion sensitivi label
        # if not int(l) in emotion_sensitive_label:
        #     continue

        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        # print(name, name1, name2, name3, name4)
        assert name == name1 == name2 == name3 == name4
        mean_e += r11.mean()
        score_e = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()
        # print(len(score))
        
        # rank_5 = score.argsort()[-5:]

        # Motion enhance
        score_emotion = enhance_emotion(score_e, emotions[name], l)

        stat_list = []
        stat_list.append(name)
        stat_list.append(l)
        stat_list.append(f"{score_diff(score_emotion, int(l)):.4f}")
        stat_list.append(f"{score_diff_with_softmax(score_emotion, int(l)):.4f}")
        
        for i in emotions[name][1]:
             stat_list.append(f"{i:.10f}")

        for i in score_emotion:
             stat_list.append(f"{i:.10f}")

        with open(SAVE_PATH_FOLDER + "/stat.csv", 'a', newline='\n') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(stat_list)


        rank_5_e = score_emotion.argsort()[-5:]
        # if not np.array_equal(rank_5, rank_5_e):
        #     print(f"Different prediction {name}, original {l}")
        #     print(f"Emotion {emotions[name][1]}")
        #     print(f"Normal: {rank_5}")
        #     print(f"Emotion: {rank_5_e}")

        # Save
        # print(rank_5_e)
        right_num_5_e += int(int(l) in rank_5_e)
        r_e = np.argmax(score_emotion)
        scores_e.append(score_emotion)
        preds_e.append(r_e)
        right_num_e += int(r_e == int(l))
        total_num_e += 1
        f.write('{}, {}\n'.format(name, r_e))

        # new_score = score_diff(score, int(l))
        # print(new_score)
        new_losses.append(score_diff(score_emotion, int(l)))
        new_losses_softmax.append(score_diff_with_softmax(score_emotion, int(l)))


    acc_e = right_num_e / total_num_e
    acc5_e = right_num_5_e / total_num_e
    print(total_num_e)
    print('top1: ', acc_e)
    print('top5: ', acc5_e)
    print(f'Loss: {statistics.mean(new_losses)}')
    print(f'Loss softmax: {statistics.mean(new_losses_softmax)}')

f.close()
f_w_to_e.close()
print(mean_e/len(label[0]))

with open(SAVE_PATH_FOLDER + '/val_pred.pkl', 'wb') as f:
    # score_dict = dict(zip(names, preds))
    score_dict = (names, preds_e)
    pickle.dump(score_dict, f)

with open(SAVE_PATH_FOLDER + '/gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores_e))
    pickle.dump(score_dict, f)