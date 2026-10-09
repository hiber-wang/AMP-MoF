import json
import cv2
import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.transform import Rotation as R
from einops import rearrange
import imageio
from PIL import Image
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
print(root_path)
sys.path.append(root_path)
from dataset.datasets_utils import reverse_seq_data, get_meta_data
# import debugpy; debugpy.connect(("10.34.8.81", 5999))

def quaternion_to_rotation_matrix(q):

    q_w, q_x, q_y, q_z = q
    
    R = np.array([
        [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y)],
        [2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x)],
        [2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)]
    ])
    return R

def create_transformation_matrix(rotation, translation):

    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

class TrainDataset(Dataset):
    def __init__(self, nuscenes_path, nuscenes_json_path, condition_frames=9, downsample_fps=3, downsample_size=16, h=256, w=512):
        self.token_path_data = []
        self.pose_data = []
        self.nuscenes_path = nuscenes_path
        self.condition_frames = condition_frames
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 10
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_token_path = []
            tmp_pose = []
            # tmp_feature_path = []
            token_path_poses = nuscenes_preprocess_data[video_keys]
            if len(token_path_poses) <= self.condition_frames * self.downsample:
                continue
            for token_path_pose in token_path_poses: 
                tmp_token_path.append(os.path.join(nuscenes_path, token_path_pose['data_path']))
                #tmp_feature_path.append(os.path.join(nuscenes_path, 'feature', os.path.splitext(os.path.basename(token_path_pose['data_path']))[0]+'.npy'))
                tmp_pose.append(token_path_pose['ego_pose'])
            self.token_path_data.append(tmp_token_path)
            # self.feature_path_data.append(tmp_feature_path)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        

    def __len__(self):
        return len(self.token_path_data)
    
    def loadarray_single(self, rotation, translation):
        absolute_transforms = np.zeros((4, 4))
        absolute_transforms[3, 3] = 1
        absolute_transforms[:3, :3] = R.from_quat(rotation).as_matrix()
        absolute_transforms[:3, 3] = translation
        return absolute_transforms
    

    def get_token_feature(self, index):
        token_paths = self.token_path_data[index]
        # feature_paths = self.feature_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(token_paths)
        start = 0 # random.randint(0, clip_length-(self.condition_frames+1)*self.downsample)
        tokens = []
        # features = []
        poses_new = []
        for i in range(self.condition_frames+1):
            try: 
                im = cv2.cvtColor(cv2.imread(token_paths[start+i*self.downsample]), cv2.COLOR_BGR2RGB)
            except:
                # in case image does not exist
                print(f'!!!! Warning:  not exists!')
                return None, None
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = cv2.resize(im, (w_1, self.h))
            else:
                h_1 = round(h / w * self.w)
                im = cv2.resize(im, (self.w, h_1))
            # token = rearrange(token, 'n (h w) -> n h w', 
            # h=self.h // self.downsample_size, w=self.w // self.downsample_size)
            tokens.append(np.array(im))
            # feature = np.load(feature_paths[start+i*self.downsample])
            # feature = rearrange(feature, 'n (h w) c -> n h w c', 
            # h=self.h // self.downsample_size, w=self.w // self.downsample_size)
            # features.append(feature)
            # pose1 = self.__loadarray_single(poses[start+i*self.downsample]["rotation"], poses[start+i*self.downsample]["translation"])
            rotation = quaternion_to_rotation_matrix(poses[start+i*self.downsample]["rotation"])
            pose2 = create_transformation_matrix(rotation, poses[start+i*self.downsample]["translation"])
            # poses_new.append(self.loadarray_single(poses[start+i*self.downsample]["rotation"], poses[start+i*self.downsample]["translation"]))
            poses_new.append(pose2)
        #pose_dict = get_meta_data(poses=poses_new)
        poses_yaw = np.array(poses_new)
        return tokens, poses_yaw

    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        if iw == self.w:
            x = int(ih/2-self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2-self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs
    
    def __getitem__(self, index):
        # tokens, poses = self.get_token_feature(index)
        while True:
            tokens, poses = self.get_token_feature(index)
            if (tokens is not None) and (poses is not None) :
                break
            else:
                index += random.randint(-index+1, self.__len__() - index - 1)
        tokens = self.aug_seq(tokens)
        tokens_tensor = []
        # features_tensor = []
        poses_tensor = []
        # yaws_tensor = []
        length = len(tokens)
        for i in range(length):
            tokens_tensor.append(torch.from_numpy(tokens[i].copy()).permute(2, 0, 1))
            # features_tensor.append(torch.from_numpy(features[i].copy()))
            poses_tensor.append(torch.from_numpy(poses[i].copy()))
            # yaws_tensor.append(torch.from_numpy(yaws[i].copy()))
        imgs = self.normalize_imgs(torch.stack(tokens_tensor, 0))
        return imgs,  torch.stack(poses_tensor, 0).float()

class ValDataset(Dataset):
    def __init__(self, nuscenes_path, nuscenes_json_path, condition_frames=3, downsample_fps=3, downsample_size=16, h=256, w=512, target_frame = -5):
        self.token_path_data = []
        self.pose_data = []
        self.feature_path_data = []
        self.target_frame = target_frame
        assert self.target_frame < 0
        # load nuscenes
        self.nuscenes_path = nuscenes_path
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12 # 10 hz
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_token_path = []
            tmp_pose = []
            tmp_feature_path = []
            token_path_poses = nuscenes_preprocess_data[video_keys]
            for token_path_pose in token_path_poses:
                tmp_token_path.append(os.path.join(nuscenes_path, 'token', os.path.splitext(os.path.basename(token_path_pose['data_path']))[0]+'.npy'))
                tmp_feature_path.append(os.path.join(nuscenes_path, 'feature', os.path.splitext(os.path.basename(token_path_pose['data_path']))[0]+'.npy'))
                tmp_pose.append(token_path_pose['ego_pose'])
            self.token_path_data.append(tmp_token_path)
            self.feature_path_data.append(tmp_feature_path)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.condition_frames = condition_frames

    def __len__(self):
        return len(self.token_path_data)    

    def get_token_feature(self, index):
        token_paths = self.token_path_data[index]
        feature_paths = self.feature_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(token_paths)
        
        target_index = clip_length + self.target_frame

        start_index = target_index - self.downsample * (self.condition_frames + 1)

        tokens = []
        features = []
        poses_new = []
        for i in range(self.condition_frames + 1):
            token = np.load(token_paths[start_index + i*self.downsample])
            # token = rearrange(token, 'n (h w) -> n h w', 
            # h=self.h // self.downsample_size, w=self.w // self.downsample_size)
            tokens.append(token)
            feature = np.load(feature_paths[start_index + i*self.downsample])
            # feature = rearrange(feature, 'n (h w) c -> n h w c', 
            # h=self.h // self.downsample_size, w=self.w // self.downsample_size)
            features.append(feature)
            poses_new.append(poses[start_index+i*self.downsample])
        pose_dict = get_meta_data(poses=poses_new)
        return tokens, features, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
    def __getitem__(self, index):
        tokens, features, poses, yaws = self.get_token_feature(index)
        tokens_tensor = []
        features_tensor = []
        poses_tensor = []
        yaws_tensor = []
        length = len(tokens)
        for i in range(length):
            tokens_tensor.append(torch.from_numpy(tokens[i].copy()))
            features_tensor.append(torch.from_numpy(features[i].copy()))
            poses_tensor.append(torch.from_numpy(poses[i].copy()))
            yaws_tensor.append(torch.from_numpy(yaws[i].copy()))
        return torch.cat(tokens_tensor, 0),  torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()






class TrainImgDataset(Dataset):
    def __init__(
        self, nuscenes_path, nuscenes_json_path, condition_frames=3, downsample_fps=3, downsample_size=16, h=256, w=512,
        reverse_seq=False
    ):
        self.img_path_data = []
        self.pose_data = []
        # load nuscenes
        self.nuscenes_path = nuscenes_path
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12 # 10 hz
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_img_path = []
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            for img_path_pose in img_path_poses:
                tmp_img_path.append(os.path.join(nuscenes_path, img_path_pose['data_path']))
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_img_path)
            self.pose_data.append(tmp_pose)
        self.img_path_data = self.img_path_data * 10
        self.pose_data = self.pose_data * 10
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.condition_frames = condition_frames
        self.reverse_seq = reverse_seq
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)

    def __len__(self):
        return len(self.img_path_data)    
    
    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        assert self.h == 256, self.w == 512
        if iw == 512:
            x = int(ih/2-self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2-self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs   
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def getimg(self, index):
        img_paths = self.img_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(img_paths)
        start = random.randint(0, clip_length-(self.condition_frames+1)*self.downsample)
        ims = []
        poses_new = []
        for i in range(self.condition_frames+1):
            im = cv2.cvtColor(cv2.imread(img_paths[start+i*self.downsample]), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = cv2.resize(im, (w_1, self.h))
            else:
                h_1 = round(h / w * self.w)
                im = cv2.resize(im, (self.w, h_1))
            ims.append(im)
            poses_new.append(poses[start+i*self.downsample])
        pose_dict = get_meta_data(poses=poses_new)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
    def __getitem__(self, index):
        imgs, poses, yaws = self.getimg(index)
        
        imgs = self.aug_seq(imgs)
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        for img, pose, yaw in zip(imgs, poses, yaws):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()

class ValImgDataset(Dataset):
    def __init__(self, nuscenes_path, nuscenes_json_path, condition_frames=3, downsample_fps=3, downsample_size=16, h=256, w=512, target_frame = -5):
        self.img_path_data = []
        self.pose_data = []
        self.target_frame = target_frame
        assert self.target_frame < 0
        # load nuscenes
        self.nuscenes_path = nuscenes_path
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12 # 10 hz
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_img_path = []
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            for img_path_pose in img_path_poses:
                tmp_img_path.append(os.path.join(nuscenes_path, img_path_pose['data_path']))
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_img_path)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.condition_frames = condition_frames
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)

    def __len__(self):
        return len(self.img_path_data)    
    
    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        assert self.h == 256, self.w == 512
        if iw == 512:
            x = int(ih/2-self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2-self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs   
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def getimg(self, index):
        img_paths = self.img_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(img_paths)
        target_index = clip_length + self.target_frame

        start_index = target_index - self.downsample * (self.condition_frames + 1)
        ims = []
        poses_new = []
        for i in range(self.condition_frames+1):
            im = cv2.cvtColor(cv2.imread(img_paths[start_index+i*self.downsample]), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = cv2.resize(im, (w_1, self.h))
            else:
                h_1 = round(h / w * self.w)
                im = cv2.resize(im, (self.w, h_1))
            ims.append(im)
            poses_new.append(poses[start_index+i*self.downsample])
        pose_dict = get_meta_data(poses=poses_new)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
    def __getitem__(self, index):
        imgs, poses, yaws = self.getimg(index)
        
        imgs = self.aug_seq(imgs)
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        for img, pose, yaw in zip(imgs, poses, yaws):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()


class TestImgDataset(Dataset):
    def __init__(self, nuscenes_path, nuscenes_json_path, condition_frames=3, downsample_fps=3, downsample_size=16, h=256, w=512):
        self.img_path_data = []
        self.pose_data = []
        # load nuscenes
        self.nuscenes_path = nuscenes_path
        with open(nuscenes_json_path, 'r', encoding='utf-8') as file:
            nuscenes_preprocess_data = json.load(file)
        self.ori_fps = 12 # 10 hz
        self.downsample = self.ori_fps // downsample_fps
        nuscenes_keys = sorted(list(nuscenes_preprocess_data.keys()))
        for video_keys in nuscenes_keys:
            tmp_img_path = []
            tmp_pose = []
            img_path_poses = nuscenes_preprocess_data[video_keys]
            for img_path_pose in img_path_poses:
                tmp_img_path.append(os.path.join(nuscenes_path, img_path_pose['data_path']))
                tmp_pose.append(img_path_pose['ego_pose'])
            self.img_path_data.append(tmp_img_path)
            self.pose_data.append(tmp_pose)
        self.h = h
        self.w = w
        self.downsample_size = downsample_size
        self.condition_frames = condition_frames
        print("self.downsample_size, self.condition_frames, self.downsample_fps", self.downsample_size, self.condition_frames, downsample_fps)

    def __len__(self):
        return len(self.img_path_data)    
    
    def aug_seq(self, imgs):
        ih, iw, _ = imgs[0].shape
        assert self.h == 256, self.w == 512
        if iw == 512:
            x = int(ih/2-self.h/2)
            y = 0
        else:
            x = 0
            y = int(iw/2-self.w/2)
        for i in range(len(imgs)):
            imgs[i] = imgs[i][x:x+self.h, y:y+self.w, :]
        return imgs   
    
    def normalize_imgs(self, imgs):
        imgs = imgs / 255.0
        imgs = (imgs - 0.5)*2
        return imgs

    def getimg(self, index):
        img_paths = self.img_path_data[index]
        poses = self.pose_data[index]
        clip_length = len(img_paths)//self.downsample
        ims = []
        poses_new = []
        for i in range(clip_length):
            im = cv2.cvtColor(cv2.imread(img_paths[i*self.downsample]), cv2.COLOR_BGR2RGB)
            h, w, _ = im.shape
            if 2*h < w:
                w_1 = round(w / h * self.h)
                im = cv2.resize(im, (w_1, self.h))
            else:
                h_1 = round(h / w * self.w)
                im = cv2.resize(im, (self.w, h_1))
            ims.append(im)
            poses_new.append(poses[i*self.downsample])
        pose_dict = get_meta_data(poses=poses_new)
        return ims, pose_dict['rel_poses'], pose_dict['rel_yaws']
            
    def __getitem__(self, index):
        imgs, poses, yaws = self.getimg(index)
        
        imgs = self.aug_seq(imgs)
        imgs_tensor = []
        poses_tensor = []
        yaws_tensor = []
        for img, pose, yaw in zip(imgs, poses, yaws):
            imgs_tensor.append(torch.from_numpy(img.copy()).permute(2, 0, 1))
            poses_tensor.append(torch.from_numpy(pose.copy()))
            yaws_tensor.append(torch.from_numpy(yaw.copy()))
        imgs = self.normalize_imgs(torch.stack(imgs_tensor, 0))
        return imgs, torch.stack(poses_tensor, 0).float(), torch.stack(yaws_tensor, 0).float()