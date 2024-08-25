import numpy as np
import torch
import argparse
import yaml
import os
import pickle
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

"""
This file create save and manage the feature embeddings of SL-GCN 
"""

def init_seed(_):
    torch.cuda.manual_seed_all(1)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing the embeddings ')

    parser.add_argument('-Experiment_name', default='')
    parser.add_argument('-output_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=12,
        help='the number of worker for data loader')
    parser.add_argument(
        '--feeder-args',
        default=dict(),
        help='the arguments of data loader')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--keep_rate',
        type=float,
        default=0.9,
        help='keep probability for drop')
    parser.add_argument(
        '--groups',
        type=int,
        default=8,
        help='decouple groups')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)

    # retrieval
    parser.add_argument('--embeddings', default='')
    parser.add_argument('--target_file', default='')
    parser.add_argument('--label_file', default='')
    parser.add_argument('--out_folder', default='')
    parser.add_argument('--part', default='')
    return parser


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    


class Video2Vec:
    """ 
        Processor for Skeleton-based Action Recgnition Embedding rappresentation
    """
    def __init__(self, arg):

        arg.model_saved_name = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Retrival/Embeddings/model/" + arg.Experiment_name 
        arg.work_dir = "/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/Retrival/Embeddings/" + arg.Experiment_name

        if not os.path.exists(arg.work_dir):
            os.mkdir(arg.work_dir)

        self.arg = arg
        self.dataset = {}
        self.save_arg()

        self.load_model()
    


    def load_model(self):
        """
        load the SL-GCN model
        """
        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
    
        if self.arg.weights:
            print('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    print('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)
                
    

    def embed_dataset(self, source_loader=['data']):
        """
        convert the dataset in a embedding space
        """
        npy = np.load(self.arg.feeder_args['data_path'])
    

        labels = open(self.arg.feeder_args['label_path'], 'rb')
        labels = np.array(pickle.load(labels))

        for i in tqdm(range(labels.shape[1])):
            skels = npy[i]
            
            
            name, l = labels[:, i]


            # pre proessing
            skels = torch.tensor(skels, dtype=torch.float32)
            skels = skels.unsqueeze(0)
            # print(f'data.shape:{skels.shape}, data_type:{type(skels)} , data_dtype:{skels.dtype}')

            skels = Variable(
                        skels.float().cuda(self.output_device),
                        requires_grad=False)

        
            with torch.no_grad():
                    embed = self.model.embed(skels)
                    #print(f'embed.size:{embed.size()}')

                    
                    # save the label and the tensor  
                    self.dataset[name] = {'label':l}
                    self.dataset[name]['tensor'] = embed
                    #print(f'finish load: {self.dataset}')

    
    def save_dataset(self):
        """
        save the embedded dataset
        """

        # convert embeddings to dict
        data = {"model": self.arg.Experiment_name, "embeddings": self.dataset}

        torch.save(
            data, os.path.join(self.arg.work_dir, self.arg.output_name)
        )

    def load_dataset(self, source):
        """
        load the embedded dataset
        """
        data = torch.load(source)

        self.dataset = data['embeddings']
        print(f'loading embeddings dataset from {source}, {len(processor.dataset)}')

    

    def embed_video(self, skel):
        """
        embedded a specific skeleton keypoints tensor
        """
        # pre proessing
        skel = torch.tensor(skel, dtype=torch.float32)
        skel = skel.unsqueeze(0)

        #print(f'data.shape:{skel.shape}, data_type:{type(skel)} , data_dtype:{skel.dtype}')

        skel = Variable(
                    skel.float().cuda(self.output_device),
                    requires_grad=False)
    
        with torch.no_grad():
                embed = self.model.embed(skel)
        
        return embed


        
    
    def similar_videos(self, target_file, label_file, out_folder, partition, num_classes=2000, n=None):
        """
        Function for comparing target video to embedded videos dataset

        Parameters:
        -----------
        target_video: string specifying the path of target file .npy to compare
            with the saved feature embedding dataset
        label_file: string specifying the path of file .pkl
        type: the type of selection for the target score  
        n: int specifying the top n most similar videos to return
        """
       
        # load data and label 
        npy = np.load(target_file)

        labels = open(label_file, 'rb')
        labels = np.array(pickle.load(labels))

        # create the vectors 
        final_vec = np.zeros((npy.shape[0],3,num_classes), dtype=np.float32)
        
        
        names =  ['top1', 'top10','mean_all']

        for part,filename in enumerate(names):
            # init the dict
            out_dict = {}
            
            for i in tqdm(range(npy.shape[0])):
                print(labels[0][i])
                name= labels[0][i]
                target_vec = self.embed_video(npy[i])


                # initiate computation of consine similarity
                cosine = nn.CosineSimilarity(dim=1)

                # iteratively store similarity of stored images to target image
                sim_dict = {}
                for k, v in self.dataset.items():
                    
                    if k == name:
                        print(f'sample {name} can not be store in the dict, because he his the target')
                        continue
                    sim = cosine(v['tensor'], target_vec)[0].item()
                    sim_dict[k] = sim

                # sort based on decreasing similarity
                items = sim_dict.items()
                sim_dict = {k: {'similarity':v} for k, v in sorted(items, key=lambda i: i[1], reverse=True)}


                # create the vectors 
                #final_vec = np.zeros((5,num_classes), dtype=np.float32)



                # do the 1hot encoder vector 
                
                if filename == 'top1':
                    # cut to defined top 1 video
                    sim_dict_c = dict(list(sim_dict.items())[:1])
                    for k in sim_dict_c.keys() & self.dataset.keys():
                        l_i = int(self.dataset[k]['label'])
                    final_vec[i,0,l_i] = 1

                if filename == 'top5':
                    # do the top 5 video
                    sim_dict_c = dict(list(sim_dict.items())[:5])
                    for k in sim_dict_c.keys() & self.dataset.keys():
                        l_i = int(self.dataset[k]['label'])
                        final_vec[i,1,l_i] = sim_dict_c[k]['similarity']

                if filename == 'top10':
                    # do the top 10 video
                    sim_dict_c = dict(list(sim_dict.items())[:10])
                    for k in sim_dict_c.keys() & self.dataset.keys():
                        l_i = int(self.dataset[k]['label'])
                        final_vec[i,1,l_i] = sim_dict_c[k]['similarity']

                if filename == 'max_all':
                    # fill with all the similarities
                    for k in sim_dict.keys() & self.dataset.keys():
                        l_i = int(self.dataset[k]['label'])
                        if final_vec[i,3,l_i] != 0:
                            if final_vec[i,3,l_i] < sim_dict[k]['similarity']:
                                # we change the score
                                final_vec[i,3,l_i] = sim_dict[k]['similarity']
                        else:
                            final_vec[i,3,l_i] = sim_dict[k]['similarity']

                if filename == 'mean_all':
                    # fill with all the similarities 
                    for k in sim_dict.keys() & self.dataset.keys():
                        l_i = int(self.dataset[k]['label'])
                        if final_vec[i,2,l_i] != 0:
                            # we do the mean 
                            final_vec[i,2,l_i] = (sim_dict[k]['similarity'] + final_vec[i,2,l_i]) / 2
                        else:
                            final_vec[i,2,l_i] = sim_dict[k]['similarity']

                # write in dict 
                out_dict[name] = final_vec[i,part,:]

            # write in file
            if not os.path.exists(os.path.join(out_folder, filename)):
                os.makedirs(os.path.join(out_folder, filename))
            with open('{}/{}/{}_retrieval.pkl'.format(out_folder, filename, partition), 'wb') as f:
                pickle.dump(out_dict, f)
            
            print(f'write {filename} file')
                

        


        print('finish process')

            # cut to defined top k similar videos
            #if n is not None:
            #    sim_dict = dict(list(sim_dict.items())[: int(n)])
        
            # Add labels
            #for k in sim_dict.keys() & self.dataset.keys():
            #    sim_dict[k]['label'] = self.dataset[k]['label']

        

    
    def start(self): 
        """
        starting the embedding of the dataset
        """
        self.embed_dataset()
        self.save_dataset()
        print(f'Save embeddings correctly: {len(self.dataset)}')

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir + '/eval_results')

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


if __name__ == '__main__':

     
    parser = get_parser()

     # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(0)
    print(arg)
    processor = Video2Vec(arg)
    processor.load_dataset(arg.embeddings)
    #processor.start()

    processor.similar_videos(   
                                target_file=arg.target_file,
                                label_file=arg.label_file,
                                out_folder=arg.out_folder,
                                partition=arg.part
                            )
    #print(topk_dict)

    
