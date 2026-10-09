import os
import cv2
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class DemoDataset(Dataset):
    def __init__(self, data_root, condition_frames=15, h=256, w=512):
        self.data_root = data_root
        self.video_path_list = sorted(os.listdir(self.data_root))
        self.condition_frames = condition_frames
        self.h = h
        self.w = w
    
    def __len__(self):
        return len(self.video_path_list)    
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs   
    
    def getimg(self, index):
        seq_data = self.video_path_list[index]
        video_path = os.path.join(self.data_root, seq_data)
        frames = glob.glob(os.path.join(video_path, '*.png'))
        frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        
        # !Note: Please check the poses and yaws shape
        # poses shape: [1, self.condition_frames+1, 2]
        # yaws shape: [1, self.condition_frames+1, 1]
        poses = np.load(os.path.join(video_path, 'pose.npy'))
        yaws = np.load(os.path.join(video_path, 'yaw.npy'))
        print('poses shape: ', poses.shape)
        print('yaws shape: ', yaws.shape)
        
        clip_length = len(frames)
        ims = []
        for i in range(clip_length):   
            im = Image.open(frames[i])
            im = im.resize((self.w, self.h), Image.BICUBIC)
            ims.append(np.array(im))
        return ims, poses, yaws
            
    def __getitem__(self, index):
        imgs, poses, yaws = self.getimg(index)
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        for img, pose, yaw in zip(imgs, poses, yaws):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()
    