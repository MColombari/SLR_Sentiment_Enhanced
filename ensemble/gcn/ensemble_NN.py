import csv
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import normalize

# Resources:
#       https://machinelearningmastery.com/using-dropout-regularization-in-pytorch-models/

EXP_NUMBER = '16000A' # Epoch number

LOAD_MODEL_WEIGHT_PATH = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/weights_w_retrieval/Exp6000A.pt"
PATH_EMOTIONS_TRAIN = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/train.csv"
PATH_EMOTIONS_TEST = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/test.csv"
PATH_EMOTIONS_VAL = "/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/val.csv"

PATH_SCORE_GCN_TRAIN = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/Training_data/train_dataset/gcn_ensembled.pkl"
PATH_SCORE_GCN_TEST = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/Training_data/test_dataset/gcn_ensembled.pkl"
PATH_SCORE_GCN_VAL = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/Training_data/val_dataset/gcn_ensembled.pkl"

PATH_RETRIEVAL_TRAIN = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Retrival/joint/top1/train_retrieval.pkl"
PATH_RETRIEVAL_TEST = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Retrival/joint/top1/test_retrieval.pkl"
PATH_RETRIEVAL_VAL = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Retrival/joint/top1/val_retrieval.pkl"

PATH_LABEL_TRAIN = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/train_labels.csv"
PATH_LABEL_TEST = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/test_labels.csv"
PATH_LABEL_VAL = "/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/labels/val_labels.csv"

OUTPUT_WEIGHT_PATH = f"/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/weights_w_retrieval/Exp{EXP_NUMBER}.pt"
OUTPUT_STATS_PATH = f'/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Ensemble/Ensemble_NN/weights_w_retrieval/stats/{EXP_NUMBER}_stats.csv'

PART = 'test'

SKIP_NAME = ['signer5_sample1557', 'signer11_sample59'] # Files name to skip, due to problems in dataset load.

DROP_OUT_PROB = 0.2
BATCH_SIZE = 64
num_workers = 0 
num_epochs = 10_000
num_fin = 4008
num_classes = 2000
num_hidden_1 = 2004
#num_hidden_2 = 2008
learning_rate = 0.01
weight_decay = 1e-4
# Device setting
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class MLP(nn.Module):
    def __init__(self, num_fin: int, num_hidden_1: int, num_classes: int):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_fin, num_hidden_1),
            nn.ReLU(),
            nn.Dropout(DROP_OUT_PROB),
            nn.Linear(num_hidden_1, num_classes)
        )

    def forward(self, x: torch.Tensor):
        return self.net(torch.flatten(x, 1))


def eval_acc(mlp: nn.Module, data_loader: torch.utils.data.DataLoader,
             device: torch.device, part='train'):

    if part == 'train':
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
    elif part == 'test':
        # testing 
        right_num_e = total_num_e = right_num_5_e = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                y_pred_discr = torch.argmax(y_pred, dim=1)
                acc_top1 = torch.sum((y_pred_discr == y).float())
                #print(y_pred_discr.size())
                _, y_pred_top5 =  torch.topk(y_pred, k=5, dim=1)
                #print(rank_5_e.size())
                acc_top_5 = torch.sum(y.unsqueeze(1) == y_pred_top5).float()
                
                right_num_5_e += acc_top_5
                right_num_e += acc_top1
                total_num_e += y_pred.size(0)

        acc_e = right_num_e / total_num_e
        acc5_e = right_num_5_e / total_num_e
        print(f"top1: {acc_e.item():.3f}\n"
              f"top5: {acc5_e.item():.3f}")



class InputData(Dataset):
    def __init__(self, em_path, score_path, retrieval_path, label_path):
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

        with open(retrieval_path, 'rb') as f:
                ret_file = pickle.load(f)
        
        label = {}
        with open(label_path, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                label[row[0]] = int(row[1])

        flag = False
        self.data = []
        for name, l in label.items():
            if name in SKIP_NAME:
                continue
            arr_1 = normalize([np.array(plk_file[name], dtype=np.float32)])[0]
            arr_2 = np.array(em_file[name], dtype=np.float32)
            arr_3 = normalize([np.array(ret_file[name], dtype=np.float32)])[0]
            if flag:
                print(arr_1)
                print(arr_3)
                print(type(arr_1))
                print(np.concatenate([arr_1, arr_2, arr_3]).shape)
                flag = False
            self.data.append({
                'Label': l,
                'Data': np.concatenate([arr_1, arr_2, arr_3])
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label_out = torch.tensor(self.data[idx]['Label'], device=device).long()
        data_out = torch.tensor(self.data[idx]['Data'], device=device).float()

        return data_out, label_out




data_train = InputData(em_path=PATH_EMOTIONS_TRAIN,
                        score_path=PATH_SCORE_GCN_TRAIN,
                        retrieval_path=PATH_RETRIEVAL_TRAIN,
                        label_path=PATH_LABEL_TRAIN)
data_test = InputData(em_path=PATH_EMOTIONS_TEST,
                        score_path=PATH_SCORE_GCN_TEST,
                        retrieval_path=PATH_RETRIEVAL_TEST,
                        label_path=PATH_LABEL_TEST)
data_val = InputData(em_path=PATH_EMOTIONS_VAL,
                        score_path=PATH_SCORE_GCN_VAL,
                        retrieval_path=PATH_RETRIEVAL_VAL,
                        label_path=PATH_LABEL_VAL)

dl_train = DataLoader(dataset=data_train, batch_size=BATCH_SIZE,
                      num_workers=num_workers, drop_last=True, shuffle=True)
dl_test = DataLoader(dataset=data_test, batch_size=BATCH_SIZE,
                     num_workers=num_workers, drop_last=False, shuffle=False)
dl_val = DataLoader(dataset=data_val, batch_size=BATCH_SIZE,
                     num_workers=num_workers, drop_last=False, shuffle=False)


model = MLP(num_fin, num_hidden_1, num_classes).to(device)

# # Run the model parallelly
# if torch.cuda.device_count() > 1:
#     print("Using {} GPUs".format(torch.cuda.device_count()))
#     model = nn.DataParallel(model)


loss_fun = nn.CrossEntropyLoss().to(device)
opt = SGD(model.parameters(), lr=learning_rate)


#metrics 
losses = []
accumulator = []
train_accs = []
test_accs = []
val_accs = []
# Load model weights if provided.
if LOAD_MODEL_WEIGHT_PATH:
    model.load_state_dict(torch.load(LOAD_MODEL_WEIGHT_PATH))


if PART == 'test':
    num_epochs = 1

for i in range(num_epochs):
    
    if IS_TRAIN:
        model.train()
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            accumulator.append(loss.cpu().detach().numpy()) 
            loss.backward()
            opt.step()
    
    model.eval()
    if len(accumulator) != 0:
        losses.append(np.mean(accumulator))
        accumulator.clear()
        
    train_acc = eval_acc(model, dl_train, device).cpu().detach().numpy()
    train_accs.append(train_acc)
    test_acc = eval_acc(model, dl_test, device).cpu().detach().numpy()
    test_accs.append(test_acc)
    val_acc = eval_acc(model, dl_val, device).cpu().detach().numpy()
    val_accs.append(val_acc)

    print(f"Epoch {i} train acc.: {train_acc:.3f} "
            f"test acc.: {test_acc:.3f} "
            f"val acc.: {val_acc:.3f}")
        

# Save at the end the weight and the stats 
torch.save(model.state_dict(), OUTPUT_WEIGHT_PATH)

run_details = pd.DataFrame()
run_details['train Loss'] = losses
run_details['train Accuracy'] = train_accs
run_details['test Accuracy'] = test_accs
run_details['val Accuracy'] = val_accs

run_details.to_csv(OUTPUT_STATS_PATH,index=True, index_label='epoch')
