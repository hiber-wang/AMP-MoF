import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from functools import lru_cache
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

from torchvision import transforms

class DrivingCaptures(Dataset):
   def __init__(self, root_dir, labels, is_train=True):
        self.root_dir = root_dir
        self.labels = labels
        folders = []
        with open(self.labels, "r") as fp:
          for line in fp.readlines():
               line = line.strip('\n')
               data = line.split(" ")
               folder, label = data[0], data[1]
               if label == "0":
                    folders.append(folder)
     
        self.src = []
        for folder in folders:
             files = os.listdir(os.path.join(self.root_dir, folder))
             files.sort()
             for file in files[20:]:
               self.src.append((folder, file))

        self.preprocess = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ToTensor()
            ])
     #    random.shuffle(self.src)
        n = int(0.8 * len(self.src))
        if is_train:
            self.src = self.src[:n]
        else:
            self.src = self.src[n:]

   def __len__(self):
        return len(self.src)
   
   def __getitem__(self, idx):
        folder, filename = self.src[idx]        

        img = Image.open(os.path.join(self.root_dir, folder, filename)).convert("RGB")
        img = self.preprocess(img)
        return img
    

if __name__ == "__main__":
     train_root_dir = "/home/weizi/workspace/misbehavior_prediction/src/output/behavior"
     labels = "/home/weizi/workspace/misbehavior_prediction/src/output/behavior/record.txt"
     train_set = DrivingCaptures(train_root_dir, labels)
     img= train_set[1000]
     print(img.shape)