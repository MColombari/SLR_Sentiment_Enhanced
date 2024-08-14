import argparse
import pickle
import statistics

import numpy as np
from tqdm import tqdm

import csv
import json


PATH_LABEL = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SL-GCN/sign/27/val_label.pkl'
PATH_GCN = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/prediction/gcn_ensembled.pkl"
PATH_3D_CONV = ""
PATH_SSTCN = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SSTCN/val_out.plk"
PATH_EMOTIONS = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/val.csv"
EMOTION_ASSOSIATION = "/homes/mcolombari/SLR_Sentiment_Enhanced/DAN_module/sign_emotion_assosiation.json"


label = open(PATH_LABEL, 'rb')
label = np.array(pickle.load(label))
# from gcn/ folder 
r1 = open(PATH_GCN, 'rb')
r1 = list(pickle.load(r1).items())
# from /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Conv3D/results/...
r2 = open(PATH_3D_CONV, 'rb')
r2 = list(pickle.load(r2).items())

# we don't use it 
# r3 = open('test_flow_color_w_val_finetune.pkl', 'rb')
# r3 = list(pickle.load(r3).items())

# from /work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SSTCN
r4 = open(PATH_SSTCN, 'rb')
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
    ret += ret * e_weight
    # print("after")
    # print(score[list_enhance])
    # print(score[list_reduce])
    return ret


def score_diff(score ,l):
    actuall = np.zeros(2000)
    actuall[l] = 1
    tmp = score + np.min(score)
    ret = abs((actuall - ((score-np.min(score))/(np.max(score)-np.min(score))))).sum()
    return ret


alpha = [1,0.9,0.4]  # gcn, rgb, bin_w_val, 

right_num_e = total_num_e = right_num_5_e = 0
names = []
preds_e = []
scores_e = []
mean_e = 0
new_losses = []
with open('predictions_rgb.csv', 'w') as f:

    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]

        # Only for emotion sensitivi label
        if not int(l) in emotion_sensitive_label:
            continue

        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name4, r44 = r4[i]
        assert name == name1 == name2 
        mean_e += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1]  + r44*alpha[2]) / np.array(alpha).sum() 
        score = score.squeeze()
        rank_5 = score.argsort()[-5:]

        score_emotion = enhance_emotion(score, emotions[name], l)

        rank_5_e = score_emotion.argsort()[-5:]
        if not np.array_equal(rank_5, rank_5_e):
            print(f"Different prediction {name}, original {l}")
            print(f"Emotion {emotions[name][1]}")
            print(f"Normal: {rank_5}")
            print(f"Emotion: {rank_5_e}")

        # Save
        # print(rank_5_e)
        right_num_5_e += int(int(l) in rank_5_e)
        r_e = np.argmax(score_emotion)
        scores_e.append(score_emotion)
        preds_e.append(r_e)
        right_num_e += int(r_e == int(l))
        total_num_e += 1
        f.write('{}, {}\n'.format(name, r_e))

        new_score = score_diff(score, int(l))
        # print(new_score)
        new_losses.append(new_score)
    
    acc_e = right_num_e / total_num_e
    acc5_e = right_num_5_e / total_num_e
    print(total_num_e)
    print('top1: ', acc_e)
    print('top5: ', acc5_e)
    print(f'Loss: {statistics.mean(new_losses)}')

f.close()
print(mean_e/len(label[0]))

# with open('./val_score.pkl', 'wb') as f:
#     score_dict = dict(zip(names, scores))
#     pickle.dump(score_dict, f)