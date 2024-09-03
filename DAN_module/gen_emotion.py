import csv
import math
import os
import argparse

from PIL import Image
import numpy as np
import cv2

import torch
from torchvision import transforms

from DAN.networks.dan import DAN    # Import from DAN repositorie.
from tqdm import tqdm

import pandas as pd


class Model():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']

        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load('/work/cvcs2024/SLR_sentiment_enhanced/DAN/models/affecnet8_epoch5_acc0.6209.pth',
            map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    
    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)
        
        return faces

    def fer(self, img0, no_label=False):

        # img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return -1, 0

        ##  single face detection
        x, y, w, h = faces[0]

        
        img = img0.crop((x,y, x+w, y+h))

        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            val, pred = torch.max(out,1)
            index = int(pred)
            if not no_label:    
                label = self.labels[index]
            else:
                label = index
            return label, val.item()
            # if type(label) is str:
            #     print("yes is str")
            #     return 0
            # print(f"Before {label} - {type(label)}")
            # return label



parser = argparse.ArgumentParser("gen_frames_faces")
parser.add_argument("--video_folder", type=str)
parser.add_argument("--npy_folder", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()


folder = args.video_folder
npy_folder = args.npy_folder
out_file = args.out_file


def Average(lst): 
    if len(lst) == 0:
        return 0
    return float(sum(lst) / len(lst))

def tot_sum(lst):
    sum = 0
    for em_row in lst:
        sum += Average(em_row)
    return sum

def normalize_list(lst):
    ret = []
    sum = tot_sum(lst)
    if sum == 0:
        return [0,0,0,0,0,0,0,0]
    for em_row in lst:
        ret.append(Average(em_row) / sum)
    return ret

def get_pred(model, frames):
    period = math.floor(len(frames) / 10) # Get 10% of frames.
    pred = [0,0,0,0,0,0,0,0]
    acc_list = [[],[],[],[],[],[],[],[]]
    for i in range(len(frames)):
        if (i % period) == 0:
            # register emotion
            em, acc = model.fer(Image.fromarray(frames[i]), no_label=True)
            if em == -1:
                continue
            # print(f'{em}  -  {type(em)}')
            pred[em] += 1
            acc_list[em].append(acc)
    index = int(pd.Series(pred).idxmax())
    # print(acc_list)
    # print(acc_list[index])
    accuracy = normalize_list(acc_list)
    return index, accuracy


for root, dirs, files in os.walk(folder, topdown=False):
    for name in tqdm(files):
        if 'color' in name:
            print(name)
            cap = cv2.VideoCapture(os.path.join(root, name))
            
            frames = []
            while True:
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break

            label_number, accuracy = get_pred(Model(), frames)

            print_list = [name[:-10], label_number]
            for e in accuracy:
                print_list.append(f"{e:.4f}")
            
            with open(out_file, 'a', newline='\n') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=',')
                spamwriter.writerow(print_list)

            print(f'{name[:-10]}: {label_number}')
            print(f"Accuracy: [", end='')
            for i in accuracy:
                print("{:.4f}".format(i), end=',')
            print("]")