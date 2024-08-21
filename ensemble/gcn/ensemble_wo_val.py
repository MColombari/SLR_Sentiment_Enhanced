import argparse
import csv
import pickle
import os

import numpy as np
from tqdm import tqdm

import statistics

EXPERIMENT_NUMBER = 6

PATH_JOINT = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/Ensemble_NN_train/Train_dataset/joint/epoch_0_0.9531320926009986.pkl"
PATH_JOINT_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/Ensemble_NN_train/Train_dataset/joint_motion/epoch_0_0.9505220154334998.pkl"
PATH_BONE = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/Ensemble_NN_train/Train_dataset/bone/epoch_0_0.9532455742169769.pkl"
PATH_BONE_MOTION = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/Ensemble_NN_train/Train_dataset/bone_motion/epoch_0_0.9505220154334998.pkl"

SAVE_PATH_FOLDER  = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/SL-GCN/prediction/" + str(EXPERIMENT_NUMBER)
if not os.path.exists(SAVE_PATH_FOLDER):
    os.makedirs(SAVE_PATH_FOLDER)

label = open('/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/SL-GCN/sign/27/train_label.pkl', 'rb')
label = np.array(pickle.load(label))
r1 = open(PATH_JOINT, 'rb')
r1 = list(pickle.load(r1).items())
r2 = open(PATH_JOINT_MOTION, 'rb')
r2 = list(pickle.load(r2).items())
r3 = open(PATH_BONE, 'rb')
r3 = list(pickle.load(r3).items())
r4 = open(PATH_BONE_MOTION, 'rb')
r4 = list(pickle.load(r4).items())


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def score_diff(score ,l):
    actuall = np.zeros(2000)
    actuall[l] = 1
    tmp = score + np.min(score)
    norm = (score-np.min(score))/(np.max(score)-np.min(score))
    # print(norm)
    ret = abs((actuall - norm)).sum()
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


# name, l = label[:, 2350]
# name1, r11 = r1[2350]
# name2, r22 = r2[2350]
# name3, r33 = r3[2350]
# name4, r44 = r4[2350]
# print(f"L: {name}")
# print(f"1: {name1}")
# print(f"2: {name2}")
# print(f"3: {name3}")
# print(f"4: {name4}")

# Remove redundancy
print(label.dtype)
dim = label.shape[1] - 1
skip_flag = True
new_label = np.empty((2, dim), dtype='<U21')
new_idx = 0
for i in range(len(label[0])):
    name, l = label[:, i]
    if name == 'signer8_sample1' and skip_flag:
        skip_flag = False
        continue
    new_label[0, new_idx] = name
    new_label[1, new_idx] = l
    new_idx += 1

label = new_label
print(label.shape)
# print(label)

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0
new_losses = []
new_losses_softmax = []
with open(SAVE_PATH_FOLDER + '/predictions_wo_val.csv', 'w') as f:
    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        # print(name)
        names.append(name)
        name1, r11 = r1[i]
        name2, r22 = r2[i]
        name3, r33 = r3[i]
        name4, r44 = r4[i]
        if not name == name1 == name2 == name3 == name4:
            print(f"L: {name}")
            print(f"1: {name1}")
            print(f"2: {name2}")
            print(f"3: {name3}")
            print(f"4: {name4}")
        # print(name, name1, name2, name3, name4)
        assert name == name1 == name2 == name3 == name4
        mean += r11.mean()
        score = (r11*alpha[0] + r22*alpha[1] + r33*alpha[2] + r44*alpha[3]) / np.array(alpha).sum()

        stat_list = []
        stat_list.append(name)
        stat_list.append(l)
        stat_list.append(f"{score_diff(score, int(l)):.4f}")
        stat_list.append(f"{score_diff_with_softmax(score, int(l)):.4f}")
        
        for i in score:
             stat_list.append(f"{i:.10f}")

        with open(SAVE_PATH_FOLDER + "/stat.csv", 'a', newline='\n') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(stat_list)

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
        new_losses_softmax.append(score_diff_with_softmax(score, int(l)))
    
    
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print('top1: ', acc)
    print('top5: ', acc5)
    print(f'Loss: {statistics.mean(new_losses)}')
    print(f'Loss softmax: {statistics.mean(new_losses_softmax)}')

f.close()
print(mean/len(label[0]))

with open(SAVE_PATH_FOLDER + '/val_pred.pkl', 'wb') as f:
    # score_dict = dict(zip(names, preds))
    score_dict = (names, preds)
    pickle.dump(score_dict, f)

with open(SAVE_PATH_FOLDER + '/gcn_ensembled.pkl', 'wb') as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)