import csv
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Resources:
#       https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/

LOAD_MODEL_WEIGHT_PATH = None
PATH_EMOTIONS_TRAIN = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/train.csv"
PATH_EMOTIONS_TEST = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/test.csv"
PATH_EMOTIONS_VAL = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/val.csv"

PATH_SCORE_GCN_TRAIN = ""
PATH_SCORE_GCN_TEST = ""
PATH_SCORE_GCN_VAL = ""

PATH_LABEL_TRAIN = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/train_labels.csv"
PATH_LABEL_TEST = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/test_labels.csv"
PATH_LABEL_VAL = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/val_labels.csv"

OUTPUT_WEIGHT_PATH = ""

IS_TRAIN = True

DROP_OUT_PROB = 0.2
BATCH_SIZE = 64
num_epochs = 100
num_fin = 2008
num_classes = 2000
num_hidden_1 = 2006
num_hidden_2 = 2004
learning_rate = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, num_fin: int, num_hidden_1: int,  num_hidden_2: int, num_classes: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_fin, num_hidden_1),
            nn.ReLU(),
            nn.Dropout(DROP_OUT_PROB),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2, num_classes)
        )

    def forward(self, x: torch.Tensor):
        return self.net(torch.flatten(x, 1))


def eval_acc(mlp: nn.Module, data_loader: torch.utils.data.DataLoader,
             device: torch.device):

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred_discr = torch.argmax(y_pred, dim=1)
            acc = torch.sum((y_pred_discr == y).float())
            correct += acc
            total += y_pred.size(0)

    return correct / total


class InputData(Dataset):
    def __init__(self, em_path, score_path, label_path):
        em_file = {}
        with open(em_path, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                em_file[row[0]] = \
                    [float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[5]),
                    float(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9])]
        with open(score_path, 'rb') as f:
            plk_file = pickle.load(f)
        
        label = open(label_path, 'rb')
        label = np.array(pickle.load(label))

        self.data = []
        for i in range(len(label[0])):
            name, l = label[:, i]
            arr_1 = normalize(np.array(plk_file[name]))
            arr_2 = np.array(em_file[name])
            self.data.append({
                'Label': l,
                'Data': np.concatenate(arr_1, arr_2)
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label_out = torch.tensor(self.data[idx]['Label'], device=device)
        data_out = torch.tensor(self.data[idx]['Data'], device=device)

        return data_out, label_out



data_train = InputData(em_path=PATH_EMOTIONS_TRAIN,
                        score_path=PATH_SCORE_GCN_TRAIN,
                        label_path=PATH_LABEL_TRAIN)
data_test = InputData(em_path=PATH_EMOTIONS_TEST,
                        score_path=PATH_SCORE_GCN_TEST,
                        label_path=PATH_LABEL_TEST)
data_val = InputData(em_path=PATH_EMOTIONS_VAL,
                        score_path=PATH_SCORE_GCN_VAL,
                        label_path=PATH_LABEL_VAL)

dl_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE,
                      num_workers=0, drop_last=True, shuffle=True)
dl_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE,
                     num_workers=0, drop_last=False, shuffle=False)
dl_val = DataLoader(dataset=data_val, batch_size=BATCH_SIZE,
                     num_workers=0, drop_last=False, shuffle=False)


model = MLP(num_fin, num_hidden_1, num_hidden_2, num_classes).to(device)
loss_fun = nn.CrossEntropyLoss().to(device)
opt = SGD(model.parameters(), learning_rate)


# Load model weights if provided.
if LOAD_MODEL_WEIGHT_PATH:
    model.load_state_dict(torch.load(LOAD_MODEL_WEIGHT_PATH))


if not IS_TRAIN:
    num_epochs = 1

for i in range(num_epochs):
    model.eval()
    print(f"Epoch {i} train acc.: {eval_acc(model, dl_train, device):.3f} "
            f"test acc.: {eval_acc(model, dl_test, device):.3f}"
            f"val acc.: {eval_acc(model, dl_val, device):.3f}")

    if IS_TRAIN:
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            loss.backward()
            opt.step()
        

# Save at the end the weight.
torch.save(model.state_dict(), OUTPUT_WEIGHT_PATH)
