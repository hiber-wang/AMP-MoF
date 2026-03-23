import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from functools import lru_cache
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import transforms
from video_transforms import *
from sklearn.preprocessing import MinMaxScaler

class DrivingCaptures(Dataset):
   def __init__(self, root_dir, labels, num_frames=8, is_train=True, is_cal_center=False, ratio=1):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transforms.Compose([
             GroupScale(256),
             GroupCenterCrop(224),
             Stack(roll=False),
             ToTorchFormatTensor(div=True),
             GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.labels = {}

        with open(labels, "r") as fp:
             for line in fp.readlines():
                    line = line.strip('\n')
                    data = line.split(" ")
                    folder, label = data[0], data[1]
                    self.labels[folder] = label

        files = os.listdir(self.root_dir)
        folders = []
        for file in files:
             if os.path.isdir(os.path.join(self.root_dir, file)):
                  if is_cal_center and self.labels[file]=="1":
                       continue
                  folders.append(file)
        folders.sort()
        self.src = []
        failure_folders = []
        for folder in folders:
             files = os.listdir(os.path.join(self.root_dir, folder))
             if len(files) < self.num_frames:
                  continue
             files.sort()
             if self.labels[folder] == "0":
               if num_frames != 1:
                    for file in files[:-1*self.num_frames+1]:
                         self.src.append((folder, file))
               else:
                    for file in files[:-1*self.num_frames]:
                         self.src.append((folder, file))
             else:
                 failure_folders.append(folder)

        failure_folders = random.choices(failure_folders, k=int(len(failure_folders)*ratio))
        for folder in failure_folders:
             files = os.listdir(os.path.join(self.root_dir, folder))
             if len(files) < self.num_frames:
                  continue
             files.sort()
             if num_frames != 1:
                    for file in files[len(files)-self.num_frames:len(files)-self.num_frames+1]:
                         self.src.append((folder, file))
             else:
                  for file in [files[-1]]:
                    self.src.append((folder, file))

   def __len__(self):
        return len(self.src)
   
   def __getitem__(self, idx):
        folder, init_file = self.src[idx]
        init_file = init_file.split(".")[0]
        cnt = 0
        imgs = []
        while cnt < self.num_frames:
             file = str(int(init_file) + 2*cnt)
             img_file = file + ".png"
             img = Image.open(os.path.join(self.root_dir, folder, img_file)).convert("RGB")
             imgs.append(img)
             cnt += 1

        imgs = self.transform(imgs)
        if self.num_frames == 8 or self.num_frames == 16:
          imgs = imgs.view((-1, self.num_frames) + imgs.size()[1:])

        label = int(self.labels[folder])
        label = 1 if label == 0 else -1
        label = torch.tensor(label)
        return imgs, label
       

if __name__ == "__main__":
     train_root_dir = "dataset/behavior"
     labels = "dataset/behavior/record.txt"
     train_set = DrivingCaptures(train_root_dir, labels, 8, True, False)
     imgs, label = train_set[0]
     print(label)