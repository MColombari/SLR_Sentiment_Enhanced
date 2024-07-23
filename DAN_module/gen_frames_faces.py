import cv2
import csv
import numpy as np
import torch
from torchvision import transforms
import os
from DAN.networks.dan import DAN
from PIL import Image
from tqdm import tqdm

def crop(image, center, radius, size=512):
    scale = 1.3
    radius_crop = (radius * scale).astype(np.int32)
    center_crop = (center).astype(np.int32)

    rect = (max(0,(center_crop-radius_crop)[0]), max(0,(center_crop-radius_crop)[1]), 
                 min(size,(center_crop+radius_crop)[0]), min(size,(center_crop+radius_crop)[1]))

    image = image[rect[1]:rect[3],rect[0]:rect[2],:]

    if image.shape[0] < image.shape[1]:
        top = abs(image.shape[0] - image.shape[1]) // 2
        bottom = abs(image.shape[0] - image.shape[1]) - top
        image = cv2.copyMakeBorder(image, top, bottom, 0, 0, cv2.BORDER_CONSTANT,value=(0,0,0))
    elif image.shape[0] > image.shape[1]:
        left = abs(image.shape[0] - image.shape[1]) // 2
        right = abs(image.shape[0] - image.shape[1]) - left
        image = cv2.copyMakeBorder(image, 0, 0, left, right, cv2.BORDER_CONSTANT,value=(0,0,0))
    return image



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

    def fer(self, img0):

        #img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        ##  single face detection
        x, y, w, h = faces[0]

        
        img = img0.crop((x,y, x+w, y+h))

        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label
        


selected_faces = [0,3, 4, 31] 

folder = '/work/cvcs2024/SLR_sentiment_enhanced/datasets/WLASL/WLASL/start_kit/data/train' # 'train', 'test'
npy_folder = '/work/cvcs2024/SLR_sentiment_enhanced/SLRSE_model_data/data-prepare/demo/train_npy' #'val_npy/npy3' # 'train_npy/npy3', 'test_npy/npy3'
out_file= '/work/cvcs2024/SLR_sentiment_enhanced/DAN/results/train.csv' 


dict_emotion= {'neutral': 0,
               'happy': 1,
               'sad': 2,
               'surprise': 3,
               'fear': 4,
               'disgust':5,
               'anger':6,
               'contempt':7 
               }



for root, dirs, files in os.walk(folder, topdown=False):
    for name in tqdm(files):
        if 'color' in name:
            if  not os.path.exists(os.path.join(npy_folder, name + '.npy')):
                continue
            cap = cv2.VideoCapture(os.path.join(root, name))
            npy = np.load(os.path.join(npy_folder, name + '.npy')).astype(np.float32)
            npy = npy[:, selected_faces, :2]
            nose_point = np.mean(npy[:,0,:],axis=0)
            # npy[:, :, 0] = 512 - npy[:, :, 0]
            xy_max = npy.max(axis=1, keepdims=False).max(axis=0, keepdims=False)
            xy_min = npy.min(axis=1, keepdims=False).min(axis=0, keepdims=False)
            assert xy_max.shape == (2,)
            xy_center = nose_point # - 20 why?!?!?!?!?
            
        
            # compute diff
            diff_tensor = nose_point - npy
        
            xy_radius = np.max(np.linalg.norm(diff_tensor,axis=2))

            index = 0
            while True:
                ret, frame = cap.read()
                if ret:
                    height, width, channels = frame.shape

                    # print(f'image size {height}, {width}')

                    frame = cv2.resize(frame, (256,256))

                    # img = plot_31_pose(frame, xy_center, npy[index])
                    # cv2.imwrite(os.path.join(output_test_folder,'{:04d}_non_crop.jpg'.format(index+1)), img)
                    # print(os.path.join(out_folder, name[:-10], '{:04d}_non_crop.jpg'.format(index+1)))

                    # print(f'xy_max: {xy_max}, xy_min:{xy_min}')
                    # print(f'center: {xy_center}')
                    # print(f'radius: {xy_radius}')
                    # print(frame.shape)
                    
                    
                    image = crop(frame, xy_center, xy_radius)
                else:
                    break
                index = index + 1
                image = cv2.resize(image, (256,256))

                model = Model()
                
                label = model.fer(Image.fromarray(image))

                label_number = dict_emotion[label]

                with open(out_file, 'w', newline='\n') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=',')
                    spamwriter.writerow([name[:-10],label_number])
            
                #print(os.path.join(out_folder, name[:-10], '{:04d}.jpg'.format(index)))
            
