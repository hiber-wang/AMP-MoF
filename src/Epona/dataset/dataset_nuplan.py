import json
import cv2
import os
import math
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R
import shutil
from PIL import Image
from utils.comm import _is_free_port, _find_free_port, _init_dist_envi

def radians_to_degrees(radians):
    degrees = radians * (180 / math.pi)
    return degrees

def get_meta_data(poses, condition_frames):
    poses = np.concatenate([poses[0:1], poses], axis=0)
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    xyzs = rel_pose[:, :3, 3]
    xys = xyzs[:, :2]
    rel_yaws = radians_to_degrees(R.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0])[:, np.newaxis]

    return {
        'rel_poses': xys,
        'rel_yaws': rel_yaws,
    }

class NuPlan(Dataset):
    def __init__(self, data_root, json_root, cache_path=None, vae=None, split='train', condition_frames=3, block_size=1, downsample_fps=3, downsample_size=16, h=256, w=512, no_pose=False, clip_num=10000, augmenter=None, paug=0.9):
        self.split = split
        if split == 'train':
            self.meta_path = f'{json_root}/train_meta.json'
            self.pose_meta_path = f'{json_root}/ego_meta'
        elif split == 'test':
            self.meta_path = f'{json_root}/test_meta.json'
            self.pose_meta_path = f'{json_root}/test_ego_meta'

        self.condition_frames = condition_frames
        self.block_size = block_size
        self.data_root = data_root
        self.cache_path = cache_path
        self.vae = vae
        
        # preprocess training list
        # self.load_preprocess = True if os.path.exists(self.meta_path) else False
        
        self.ori_fps = 10 # original freq is 10 hz
        self.downsample = self.ori_fps // downsample_fps
        self.h = h
        self.w = w
        self.no_pose = no_pose

        # load preprocessed meta
        with open(self.meta_path, 'r') as f:
            json_data = json.load(f)

        # self.sequences = json_data
        json_data_filter = []
        for data in json_data:
            if len(data['CAM_F0']) > self.condition_frames * self.downsample:
                json_data_filter.append(data)
        
        self.sequences = json_data_filter

        self.downsample_size = downsample_size
        
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)
    
    def __len__(self):
        return len(self.sequences)    
    
    def load_pose(self, pose_path, front_cam_list):
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        front_cam_pose = pose_data['CAM_F0'] if 'CAM_F0' in pose_data else pose_data
        poses = {key:
            [pose_meta['x'],pose_meta['y'],pose_meta['z'],
            pose_meta['qx'],pose_meta['qy'],pose_meta['qz'], pose_meta['qw'],] for key, pose_meta in front_cam_pose.items()}
        poses_filter = np.array([poses[f"CAM_F0/{ts}"] for ts in front_cam_list])
        return poses_filter
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def __loadarray_tum_single(self, array):
        absolute_transforms = np.zeros((4, 4))
        absolute_transforms[3, 3] = 1
        absolute_transforms[:3, :3] = R.from_quat(array[3:7]).as_matrix()
        absolute_transforms[:3, 3] = array[0:3]
        return absolute_transforms
        
    def downsample_sequences(self, img_ts, poses):
        ori_size = len(img_ts)
        assert len(img_ts) == len(poses)
        index_list = np.arange(0, ori_size, step=self.downsample)
        img_ts_downsample =np.array(img_ts)[index_list]
        poses_downsample = poses[index_list]
        return img_ts_downsample, poses_downsample

    def getimg(self, index):
        seq_data = self.sequences[index]

        seq_root = os.path.join(self.data_root, seq_data['data_root'])
        seq_db_name = os.path.basename(seq_root)
        pose_path = f"{self.pose_meta_path }/{seq_db_name}.json"

        rgb_front_dir = f"{seq_root}/CAM_F0"
        try:
            poses = self.load_pose(pose_path, seq_data['CAM_F0'])
        except:
            # in case image does not exist
            # print(f'!!!! Warning: {pose_path} not exists!')
            return None, None

        # downsample fps
        img_ts_downsample, poses_downsample = self.downsample_sequences(seq_data['CAM_F0'], poses)
        clip_length = len(img_ts_downsample)
        if self.split == "val":
            start = 0
        else:
            start = random.randint(0, clip_length-self.condition_frames-self.block_size)
        ims = []
        poses_new = []
        for i in range(self.condition_frames+self.block_size):   
            try:
                im = Image.open(f"{rgb_front_dir}/{img_ts_downsample[start+i]}")
            except:
                # in case image does not exist
                print(f'!!!! Warning: {rgb_front_dir}/{img_ts_downsample[start+i]} not exists!')
                return None, None
            im = im.resize((self.w, self.h), Image.BICUBIC)
            ims.append(np.array(im))
            poses_new.append(self.__loadarray_tum_single(poses_downsample[start+i]))
        poses_yaw = np.array(poses_new)
        return ims, poses_yaw

    def __getitem__(self, index):
        while True:
            imgs, poses = self.getimg(index)
            if (imgs is not None) and (poses is not None) :
                break
            else:
                index += random.randint(-index+1, self.__len__() - index - 1)
        imgs_tensor = []
        poses_tensor = []
        # yaws_tensor = []
        for img, pose in zip(imgs, poses):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            # yaws_tensor.append(torch.from_numpy(yaw.copy()))
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        if self.no_pose:
            return imgs, torch.tensor(0.0)
        else:
            return imgs, torch.stack(poses_tensor, 0).float()
    
    def check_data(self, index, imgs, ps):
        save_dir = f'./check_train_data/{index}'
        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            print(index, i, ps[i])
            cv2.imwrite(save_dir + f'/{i}.jpg', img)