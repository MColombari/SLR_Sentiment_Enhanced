import torch
import os
import numpy as np
from T_Pose_model import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.parallel
import argparse
import pickle
from tqdm import tqdm

NUM_CLASSES = 2000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/new_test_24x24", help="Path to input dataset")
    parser.add_argument("--checkpoint_model", type=str, default="./T_Pose_model_final.pth", help="Optional path to checkpoint model")
    parser.add_argument("--test_labels", type=str, default="./test_labels_pseudo.pkl", help="path for input test labels")
    parser.add_argument("--out_file", type=str, default="./out.pkl", help="path output file")

    opt = parser.parse_args()
    print(opt)
    test_files = open(opt.test_labels, 'rb')
    test_files = np.array(pickle.load(test_files))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T_Pose_model(frames_number=60,joints_number=33,
        n_classes=NUM_CLASSES
    )
    #model = nn.DataParallel(model)    
    model = model.to(device)
    
    # Add weights from checkpoint model if specified
    if opt.checkpoint_model:
        model.load_state_dict(torch.load(opt.checkpoint_model,map_location='cuda:0'))#,strict=False)
    else:
        model.init_weights()
    model.eval()
    preds =[]
    names = []
    index = 0
    for name in tqdm(test_files):
        names.append(name)
        fea_name = name+'_color.pt'
        fea_path = os.path.join(opt.dataset_path, fea_name)
        data = torch.load(fea_path)
        data = data.contiguous().view(1,-1,24,24)
        data_in = Variable(data.to(device), requires_grad=False)
        with torch.no_grad():
             pred=model(data_in)
        pred = pred.cpu().detach().numpy()
        preds.append(pred)
        
    with open(opt.out_file, 'wb') as f:
         score_dict = dict(zip(names, preds))
         pickle.dump(score_dict, f)

