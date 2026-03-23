import json
import os
import pickle
import time
import cv2

import csv
import random

import numpy as np
from driving_model.model_car import Vgg16Speed, Vgg16Steer, Resnet101Speed, Resnet101Steer, EpochSpeed, EpochSteer
from model.autoencoder import Autoencoder
from model.abnormal_detector import AbnormalDetector
from dataset import DrivingCaptures
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import gamma
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pytorch_metric_learning import samplers, losses, miners, testers, distances, reducers
from PIL import Image
from video_transforms import *
from torchvision import transforms
from sklearn import metrics
from torchvision.utils import save_image
from DMVFN.model.model import Model as PredModel
from selforacle.detectors.deep_autoencoder import DeepAutoencoder
from collections import deque
# from visualize_pytorch.main import run
from visualize_pytorch.src.gradCAM import *
from visualize_pytorch.src.guidedBackProp import *
from visualize_pytorch.src.smoothGrad import *
torch.manual_seed(42)

score_dict = {}


class TrainRunner():
    def __init__(self, args):
        self.train_dataset = DrivingCaptures(args.root_dir, args.labels, num_frames=args.num_frames, is_train=True, ratio=args.ratio)
        self.val_dataset = DrivingCaptures(args.root_dir, args.labels, num_frames=args.num_frames, is_train=False, ratio=args.ratio)
        self.model = AbnormalDetector(args.img_size, args.num_frames, embedding_dim=args.embedding_dim).cuda()
        if args.test:
            if args.oracle != "Atten_ae":
                if args.num_frames == -8:
                    self.model.load_state_dict(torch.load(args.model+str(8)+".pyth"))
                else:
                    self.model.load_state_dict(torch.load(args.model+str(args.num_frames)+str(args.ratio)+".pyth"))
            else:
                self.init_model_from_pretraining(args)
            self.pred_model = PredModel(load_path="DMVFN/pretrained_models/dmvfn_kitti.pkl", training=False)
            self.selforacle = DeepAutoencoder(name="DAE", args=args, hidden_layer_dim=256)
            self.selforacle.initialize()
            self.selforacle.load_or_train_model(is_train=False)

            if args.oracle == "ThirdEye":
                if args.agent == "vgg16":
                    speed_model = Vgg16Speed()
                    speed_model.load_state_dict(torch.load("driving_model/model/vgg16/speed.pt"))
                    self.speed_smooth_grad = SmoothGrad(speed_model, use_cuda=True, stdev_spread=0.2, n_samples=20)

                    steer_model = Vgg16Steer()
                    steer_model.load_state_dict(torch.load("driving_model/model/vgg16/steer.pt"))
                    self.steer_smooth_grad = SmoothGrad(steer_model, use_cuda=True, stdev_spread=0.2, n_samples=20)
                    self.hd_thresholds = 0.01
                if args.agent == "resnet101":
                    speed_model = Resnet101Speed()
                    speed_model.load_state_dict(torch.load("driving_model/model/resnet101/speed.pt"))
                    self.speed_smooth_grad = SmoothGrad(speed_model, use_cuda=True, stdev_spread=0.2, n_samples=20)

                    steer_model = Resnet101Steer()
                    steer_model.load_state_dict(torch.load("driving_model/model/resnet101/steer.pt"))
                    self.steer_smooth_grad = SmoothGrad(steer_model, use_cuda=True, stdev_spread=0.2, n_samples=20)
                    self.hd_thresholds = 0.002
                
                if args.agent == "epoch":
                    speed_model = EpochSpeed()
                    speed_model.load_state_dict(torch.load("driving_model/model/epoch/speed.pt"))
                    self.speed_smooth_grad = SmoothGrad(speed_model, use_cuda=True, stdev_spread=0.2, n_samples=20)

                    steer_model = EpochSteer()
                    steer_model.load_state_dict(torch.load("driving_model/model/epoch/steer.pt"))
                    self.steer_smooth_grad = SmoothGrad(steer_model, use_cuda=True, stdev_spread=0.2, n_samples=20)
                    self.hd_thresholds = 1.3752680898260206e-07

            elif args.oracle == "Atten":
                self.R = args.R
        if args.num_frames == 1:
            self.ae_model = Autoencoder(input_dim=2048, embedding_dim=args.embedding_dim).cuda()
        else:
            self.ae_model = Autoencoder(input_dim=768, embedding_dim=args.embedding_dim).cuda()

    def pretrain(self, args):
        print("pretrain the autoencoder of control information")
        train_loader = DataLoader(self.train_dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=args.pretrain_batch_size, num_workers=args.num_workers)

        optimizer = optim.Adam(self.ae_model.parameters(), lr=args.pretrain_lr)
        criterion = nn.MSELoss()
        min_loss = float('inf')

        self.ae_model.train()
        self.model.eval()
        for epoch in range(1, 1 + args.pretrain_epochs):
            pbar = tqdm(train_loader)
            for X in pbar:
                imgs, label = X
                imgs = imgs.cuda()

                extract_features = self.model(imgs, True)
                preds = self.ae_model(extract_features)
                loss = criterion(preds, extract_features)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))
            
            with torch.no_grad():
                total_loss = 0
                for X in val_loader:
                    imgs, label = X
                    imgs = imgs.cuda()

                    extract_features = self.model(imgs, True)
                    preds = self.ae_model(extract_features)
                    loss = criterion(preds, extract_features)
                    total_loss += loss.item()
                val_loss = total_loss / len(val_loader)
                print("Epoch:", epoch, "Val loss:", val_loss, end=" ")
                if val_loss < min_loss:
                    print("update!")
                    torch.save(self.ae_model.state_dict(), args.pretrain_model+args.oracle+str(args.num_frames)+str(args.ratio)+".pyth")
                    min_loss = val_loss

        print("the pretraining of autoencoder is completed!")
    
    def init_model_from_pretraining(self, args):
        print("initlize the weight of abnormal detector......", end="")
        model_dict = self.model.state_dict()
        state_dict = torch.load(args.pretrain_model+args.oracle+str(args.num_frames)+str(args.ratio)+".pyth")
        # Filter out decoder network keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # Overwrite values in the existing state_dict
        model_dict.update(state_dict)
        # Load the new state_dict
        self.model.load_state_dict(model_dict)
        print("done!")

    def cal_center(self, args, eps=0.01):
        print("calculate the center of hypersphere")
        dataset = DrivingCaptures(args.root_dir, args.labels, num_frames=args.num_frames, is_train=True, is_cal_center=True)
        train_loader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=args.num_workers)
        self.model.eval()
        n_samples = 0
        c = torch.zeros(args.embedding_dim).cuda()
        with torch.no_grad():
            pbar = tqdm(train_loader)
            for X in pbar:
                # get the inputs of the batch
                imgs, label = X
                imgs = imgs.cuda()
                outputs = self.model(imgs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        with open(args.center_path+str(args.num_frames)+str(args.ratio)+".pkl", "wb") as fp:
            pickle.dump(c, fp)

    def train(self, args):
        print("start training")
        with open(args.center_path+str(args.num_frames)+str(args.ratio)+".pkl", "rb") as fp:
            c = pickle.load(fp)
            c = c.cuda()
        train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
        val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)

        min_loss = float("inf")
        # Training
        if args.retrain:
            self.model.load_state_dict(torch.load( args.model+str(args.num_frames)+str(args.ratio)+".pyth"))
        self.model.train()
        for epoch in range(1, args.epochs+1):
            epoch_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader)
            for X in pbar:
                imgs, label = X
                imgs = imgs.cuda()
                label = label.cuda()
                outputs = self.model(imgs)
                optimizer.zero_grad()
                dist = torch.sum((outputs - c) ** 2, dim=1)
                losses = torch.where(label == 0, dist, args.eta * ((dist + args.eps) ** label.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                scheduler.step()
                n_batches += 1
                pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))
            
            epoch_loss = epoch_loss / len(train_loader)
            print("Epoch:", epoch, "Epoch loss:",epoch_loss, end=" ")
            if epoch_loss < min_loss:
                print("update!")
                torch.save(self.model.state_dict(), args.model+str(args.num_frames)+str(args.ratio)+".pyth")
                min_loss = epoch_loss

        print("training is completed!")


    def cal_assert_ae(self, args):
        with open(args.center_path+str(args.num_frames)+str(args.ratio)+".pkl", "rb") as fp:
            c = pickle.load(fp)
            c = c.cuda()
        print("calculate the center of hypersphere")
        dataset = DrivingCaptures(args.root_dir, args.labels, num_frames=args.num_frames, is_train=True, is_cal_center=True)
        train_loader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=args.num_workers)
        self.model.eval()
        loss_list = []
        with torch.no_grad():
            pbar = tqdm(train_loader)
            for X in pbar:
                # get the inputs of the batch
                imgs, label = X
                imgs = imgs.cuda()
                outputs = self.model(imgs)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                outputs_list= dist.cpu().numpy().tolist()
                loss_list += outputs_list
        loss_list = np.array(loss_list)
        shape, loc, scale = gamma.fit(loss_list, floc=-0.1)
        thresholds = {}
        conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
        for c in conf_intervals:
            thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)
        as_json = json.dumps(thresholds)

        assert_thresholds = os.path.join("output", "thresholds_"+str(args.oracle)+str(args.agent)+str(args.num_frames)+str(args.ratio)+".json")
        print("Saving thresholds to %s" % assert_thresholds)
        with open(assert_thresholds, 'w') as fp:
            fp.write(as_json)
    
    def calc_and_store_thresholds(self, args):
        self.model.load_state_dict(torch.load( args.model+str(args.num_frames)+str(args.ratio)+".pyth"))
        with open(args.center_path+str(args.num_frames)+str(args.ratio)+".pkl", "rb") as fp:
            c = pickle.load(fp)
            c = c.cuda()
        train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
        loss_list = []
        with torch.no_grad():
            total_loss = 0
            pbar = tqdm(train_loader)
            for X in pbar:
                imgs, label = X
                imgs = imgs.cuda()
                label = label.cuda()
                outputs = self.model(imgs)
                dist = torch.sum((outputs - c) ** 2, dim=1)
                losses = torch.where(label == 0, dist, args.eta * ((dist + args.eps) ** label.float()))
                loss = torch.mean(losses)
                pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))
                loss_list += losses.cpu().numpy().tolist()


        loss_list = np.array(loss_list)
        shape, loc, scale = gamma.fit(loss_list, floc=-0.1)
        thresholds = {}
        conf_intervals = [0.68, 0.90, 0.95, 0.99, 0.999, 0.9999, 0.99999]
        for c in conf_intervals:
            thresholds[str(c)] = gamma.ppf(c, shape, loc=loc, scale=scale)
        as_json = json.dumps(thresholds)

        assert_thresholds = os.path.join("output", "thresholds_"+str(args.num_frames)+str(args.ratio)+".json")
        print("Saving thresholds to %s" % assert_thresholds)
        with open(assert_thresholds, 'w') as fp:
            fp.write(as_json)
    
    def  test_for_agent(self, args):
        global score_dict

        print("Agent:", args.agent, "Oracle:", args.oracle)
        if args.oracle == "Atten":
            print("R:", self.R)
        tp, fp, tn, fn = 0, 0, 0, 0

        with open(os.path.join(args.output,  str(args.oracle)+"_"+str(args.agent), "record.txt"), "r") as f:
            folders = f.readlines()
            random.shuffle(folders)
            pbar = tqdm(folders)
            for line in pbar:
                line = line.strip('\n')
                data = line.split(" ")
                folder, label = data[0], data[1]
                folder_path = os.path.join(args.output,   args.agent, folder)
                if folder_path in score_dict.keys():
                    continue
                frames = os.listdir(folder_path)
                frames = list(frames)
                frames.sort(key=lambda x: int(x[:-4]))
                if label == "0":
                    warning_frames = self.test_for_single_folder(args, folder_path)
                    frame_lists = [frames[i:i+30] for i in range(0, len(frames), 30)]
                    for frame_list in frame_lists:
                        flag = False
                        for warning_frame in warning_frames:
                            if warning_frame in frame_list:
                                fp += 1
                                flag = True
                                break
                        if not flag:
                            tn += 1
                else:
                    warning_frames = self.test_for_single_folder(args, folder_path)
                    reaction_frames = frames[-1*args.abnormal_end:]
                    abnormal_frames = frames[-1*args.abnormal_start:-1*args.abnormal_end]
                    normal_frame_lists = [frames[i:i+30] for i in range(0, len(frames[:-1*args.abnormal_start]), 30)]
                    for normal_frame_list in normal_frame_lists:
                        flag = False
                        for warning_frame in warning_frames:
                            if warning_frame in normal_frame_list:
                                fp += 1
                                flag = True
                                break
                        if not flag:
                            tn += 1
                    
                    flag = False
                    for warning_frame in warning_frames:   
                        if warning_frame in abnormal_frames:
                            tp += 1
                            flag = True
                            break
                    if not flag:
                        fn += 1
                precision = tp / (tp+fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0
                F3 = 10*precision*recall / (9*precision+recall) if precision + recall > 0 else 0
                print("精确率:", precision, "召回率:", recall, "F3得分:", F3)
                pbar.set_description('Pr: {pr:.4f}, Re:{re:.4f}, F3:{F3:.4f}'.format(pr=precision, re=recall, F3=F3))

        print("正->正", tp, "负->负", tn, "负->正", fp, "正->负", fn)

        with open( os.path.join(args.output, args.agent+"_"+ args.oracle+str(args.num_frames)+str(args.ratio)+".pkl"), "wb") as f:
            pickle.dump(score_dict, f)



    def test_for_single_folder(self, args, folder):
        global score_dict
        score_dict[folder]=[]
        if args.num_frames == -8:
            with open(args.center_path+str(8)+".pkl", "rb") as fp:
                c = pickle.load(fp)
                c = c.cuda()
        else:
            with open(args.center_path+str(args.num_frames)+str(int(args.ratio))+".pkl", "rb") as fp:
                c = pickle.load(fp)
                c = c.cuda()
        transform = transforms.Compose([
                GroupScale(256),
                GroupCenterCrop(224),
                Stack(roll=False),
                ToTorchFormatTensor(div=True),
                GroupNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img_transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor()
            ])
        files = os.listdir(folder)
        files = list(files)
        files.sort(key=lambda x: int(x[:-4]))
        max_dist = 0
        max_deque = deque(maxlen=1)
        self_oracle_deque = deque(maxlen=5)
        warning_frames = []
        historical_window = []
        loss_fn = torch.nn.MSELoss()
        if args.oracle == "Atten":
            for idx, file in enumerate(files):
                if idx == 0:
                    continue
                init_file = file.split(".")[0]
                cnt = 0
                if args.num_frames != -8:
                    img_0 =  torch.from_numpy(cv2.imread(os.path.join(folder, files[idx-1])).astype("float32")).permute(2, 0, 1)
                    img_1 = torch.from_numpy(cv2.imread(os.path.join(folder, files[idx])).astype("float32")).permute(2, 0, 1)
                    previous_imgs = [img_0, img_1]
                    previous_imgs = torch.stack(previous_imgs, 0)
                    previous_imgs = previous_imgs.unsqueeze(0)
                    previous_imgs = previous_imgs.to("cuda", non_blocking=True) / 255.
                    start_time = time.time()
                    preds = self.pred_model.eval(previous_imgs, 'single_test', num_frames=args.num_frames)
                    transform_to_pil = transforms.ToPILImage()
                    preds_imgs = []
                    for item in preds:
                        item = item.squeeze(0)
                        item = transform_to_pil(item)
                        preds_imgs.append(item)
                    preds_imgs = transform(preds_imgs)
                    preds_imgs = preds_imgs.view((-1, args.num_frames) + preds_imgs.size()[1:])
                    if args.num_frames == 1:
                        preds_imgs = preds_imgs.squeeze(1)
                    preds_imgs = preds_imgs.cuda()
                    preds_imgs = preds_imgs.unsqueeze(0)
                else:
                    start_time = time.time()
                    img_current = Image.open(os.path.join(folder, files[idx])).convert("RGB")
                    if len(historical_window) == 8:
                        historical_window.pop(0)
                    historical_window.append(img_current)
                    if len(historical_window) < 8:
                        continue
                    preds_imgs = transform(historical_window)
                    preds_imgs = preds_imgs.view((-1, 8) + preds_imgs.size()[1:])
                    preds_imgs = preds_imgs.cuda()
                    preds_imgs = preds_imgs.unsqueeze(0)
                with torch.no_grad():
                    output = self.model(preds_imgs)
                    dist = torch.sum((output - c) ** 2, dim=1)
                    max_deque.append(dist.item())
                    max_dist = sum(max_deque) / len(max_deque)

                    if max_dist > self.R:

                        warning_frames.append(files[idx])
                end_time = time.time()
                if idx != 0:
                    score_dict[folder].append((files[idx], dist.item(), end_time - start_time))
        elif args.oracle == "Atten_ae":
            for idx, file in enumerate(files):
                if idx == 0:
                    continue
                img_0 =  torch.from_numpy(cv2.imread(os.path.join(folder, files[idx-1])).astype("float32")).permute(2, 0, 1)
                img_1 = torch.from_numpy(cv2.imread(os.path.join(folder, files[idx])).astype("float32")).permute(2, 0, 1)
                previous_imgs = [img_0, img_1]
                previous_imgs = torch.stack(previous_imgs, 0)
                previous_imgs = previous_imgs.unsqueeze(0)
                previous_imgs = previous_imgs.to("cuda", non_blocking=True) / 255.
                start_time = time.time()
                preds = self.pred_model.eval(previous_imgs, 'single_test', num_frames=args.num_frames)
                transform_to_pil = transforms.ToPILImage()
                preds_imgs = []
                for item in preds:
                    item = item.squeeze(0)
                    item = transform_to_pil(item)
                    preds_imgs.append(item)
                preds_imgs = transform(preds_imgs)
                preds_imgs = preds_imgs.view((-1, args.num_frames) + preds_imgs.size()[1:])
                if args.num_frames == 1:
                    preds_imgs = preds_imgs.squeeze(1)
                preds_imgs = preds_imgs.cuda()
                preds_imgs = preds_imgs.unsqueeze(0)

                with torch.no_grad():
                    output = self.model(preds_imgs)
                    dist = torch.sum((output - c) ** 2, dim=1)
                    max_deque.append(dist.item())
                    max_dist = sum(max_deque) / len(max_deque)
                    # print(max_dist)
                    if max_dist > 92.92009028256153:
                        warning_frames.append(files[idx])
                end_time = time.time()
                if idx != 0:
                    score_dict[folder].append((files[idx], dist.item(), end_time - start_time))

        elif args.oracle == "SelfOracle":
            for idx, file in enumerate(files):
                start_time = time.time()
                img = Image.open(os.path.join(folder, files[idx])).convert("RGB")
                X = img_transform(img).unsqueeze(0).cuda()
                batch_size = X.shape[0]
                X = X.view(batch_size, -1)
                with torch.no_grad():
                    pred = self.selforacle.model(X)
                    loss = loss_fn(X, pred)
                    self_oracle_deque.append(loss.item())
                    if sum(self_oracle_deque) / len(self_oracle_deque) > 0.023:
                        warning_frames.append(files[idx])
                end_time = time.time()
                if idx != 0:
                    score_dict[folder].append((files[idx], loss.item(), end_time - start_time))
        elif args.oracle == "ThirdEye":
            for idx, file in enumerate(files):
                start_time = time.time()
                raw_image = cv2.imread(os.path.join(folder, files[idx]))[..., ::-1]
                raw_image = cv2.resize(raw_image, (224, 224))
                image = transforms.Compose([
                    transforms.ToTensor(),
                ])(raw_image).unsqueeze(0)
                speed_smooth_cam, _ = self.speed_smooth_grad(image)
                steer_smooth_cam, _ = self.steer_smooth_grad(image)
                if idx != 0:
                     avg_smooth_cam = (abs(np.mean(speed_smooth_cam - previous_speed_smooth_cam)) + abs(np.mean(steer_smooth_cam - previous_steer_smooth_cam))) / 2
                     if avg_smooth_cam > self.hd_thresholds:
                         warning_frames.append(files[idx])
                previous_speed_smooth_cam = speed_smooth_cam
                previous_steer_smooth_cam = steer_smooth_cam
                end_time = time.time()
                if idx != 0:
                    score_dict[folder].append((files[idx], avg_smooth_cam, end_time - start_time))

        return warning_frames

def pretrain(args):
    print("pretrain the autoencoder")
    model = Autoencoder(args.seq_len, args.num_features).cuda()
    train_dataset = DrivingCaptures(args.root_dir, num_frames=args.num_frames,  is_train=True)
    val_dataset = DrivingCaptures(args.root_dir, num_frames=args.num_frames, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.pretrain_batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.pretrain_batch_size, num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=args.pretrain_lr)
    criterion = nn.MSELoss()
    min_loss = float('inf')
    for epoch in range(1, 1 + args.pretrain_epochs):
        model.train()
        pbar = tqdm(train_loader)
        for X in pbar:
            X = X[-1]
            batch_size = X.shape[0]
            X = X.cuda()
            preds = model(X)
            loss = criterion(X, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))
        
        with torch.no_grad():
            total_loss = 0
            for X in val_loader:
                X = X[-1]
                batch_size = X.shape[0]
                X = X.cuda()
                preds = model(X)
                loss = criterion(X, preds)
                total_loss += loss.item()
            val_loss = total_loss / len(val_loader)
            print("Epoch:", epoch, "Val loss:", val_loss, end=" ")
            if val_loss < min_loss:
                print("update!")
                torch.save(model.state_dict(), args.pretrain_path)
                min_loss = val_loss
    print("the pretraining of autoencoder is completed!")


def build_and_init_model_from_pretraining(args):
    print("build the network of abnormal detector......", end="")
    model = AbnormalDetector(args.img_size, args.num_frames, args.seq_len, args.num_features).cuda()
    print("done!")
    print("initlize the weight of abnormal detector......", end="")
    model_dict = model.state_dict()
    state_dict = torch.load(args.pretrain_model)
    # Filter out decoder network keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # Overwrite values in the existing state_dict
    model_dict.update(state_dict)
    # Load the new state_dict
    model.load_state_dict(model_dict)
    print("done!")
    return model

def train(args):
    model = AbnormalDetector(args.img_size, args.num_frames, args.seq_len, args.num_features, is_high_dimension=False).cuda()
    print("start training")
    train_dataset = DrivingCaptures(args.root_dir, num_frames=args.num_frames,  is_train=True, is_high_dimension=False, km=args.center_path)
    val_dataset = DrivingCaptures(args.root_dir, num_frames=args.num_frames, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # Set optimizer (Adam optimizer for now)
    distance = distances.CosineSimilarity()
    # loss_func = losses.CircleLoss(m=0.3, gamma=400, distance=distance)
    loss_func = losses.ArcFaceLoss(num_classes=20, embedding_size=256, scale=10, margin=10)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # Set learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    min_loss = float("inf")
    # Training
    model.train()
    for epoch in range(1, args.epochs+1):
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader)
        for X in pbar:
            _, front, y = X
            front = front.cuda()
            y = y.cuda()
            # Zero the network parameter gradients
            optimizer.zero_grad()
            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = model(front)

            loss = loss_func(outputs, y.long())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            scheduler.step()
            n_batches += 1
            pbar.set_description('Loss: {loss:.4f}'.format(loss=loss.item()))
        train_loss = epoch_loss / n_batches
        if train_loss < min_loss:
            torch.save(model.state_dict(), args.model)
            min_loss = train_loss
        print("Train Loss:", epoch_loss / n_batches)
    print("training is completed!")
    return model


def calc_hypersphere_center(args, eps=0.1):

    with open(args.high_dimension, "rb") as fp:
        extracted_features = pickle.load(fp)

    with open(args.idx_list, "rb") as fp:
        idx_list = pickle.load(fp)
    km = KMeans(n_clusters=50, n_init='auto', random_state=42)
    km.fit(extracted_features)
    pred_list = km.predict(extracted_features)
    ch = metrics.calinski_harabasz_score(extracted_features, pred_list)
    print(ch)
    pred_dict = {}
    for i, idx in enumerate(idx_list):
        pred_dict[idx] = pred_list[i]
    with open(args.center_path, "wb") as fp:
        pickle.dump(pred_dict, fp)
    print("calculation is completed!")


def get_embedding_library(args):
    model = AbnormalDetector(args.img_size, args.num_frames, args.seq_len, args.num_features, is_high_dimension=False)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model))
    train_dataset = DrivingCaptures(args.root_dir, args.num_frames)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    model.eval()
    extracted_features = None
    idx_list = None
    with torch.no_grad():
        pbar = tqdm(train_loader)
        for X in pbar:
            # get the inputs of the batch
            _, front = X
            front = front.cuda()
            outputs = model(front)
            if extracted_features is None:
                extracted_features = outputs
            else:
                extracted_features = torch.cat((extracted_features, outputs), dim=0)

    with open(args.embedding_library, "wb") as fp:
        pickle.dump(extracted_features, fp)


def get_distance(target, behaviored):
    distribution = torch.nn.functional.cosine_similarity(target.unsqueeze(1), behaviored.unsqueeze(0), dim=2)
    return distribution
                                 
def cal_threholds(args):
    model = AbnormalDetector(args.img_size, args.num_frames, args.seq_len, args.num_features, is_high_dimension=False)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model))
    with open(args.embedding_library, "rb") as fp:
        embedding_library = pickle.load(fp)
    test_dataset = DrivingCaptures(args.root_dir, args.num_frames, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=args.num_workers, shuffle=False)
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for X in pbar:
            _, front = X
            front = front.cuda()
            embedding_feature = model(front)
            cos_distances = get_distance(embedding_feature, embedding_library)
            similarity = torch.max(cos_distances, dim=1).values
            print(similarity)


def test(args):
    model = AbnormalDetector(args.img_size, args.num_frames, args.seq_len, args.num_features, is_high_dimension=False)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model))
    with open(args.embedding_library, "rb") as fp:
        embedding_library = pickle.load(fp)
    test_dataset = DrivingCaptures(args.root_dir, args.num_frames, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=args.num_workers, shuffle=False)
    with torch.no_grad():
        _, front = test_dataset[-2]
        front = front.cuda()
        front = front.unsqueeze(0)  
        embedding_feature = model(front)
        cos_distances = get_distance(embedding_feature, embedding_library)
        similarity = torch.max(cos_distances).item()
        print(similarity)



def test_for_single_folder(args, folder):
    model = AbnormalDetector(args.img_size, args.num_frames, args.seq_len, args.num_features, is_high_dimension=False)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model))
    transform = transforms.Compose([
             GroupScale(224),
             GroupCenterCrop(224),
             Stack(roll=False),
             ToTorchFormatTensor(div=True),
        ])
    preprocess = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    with open(args.embedding_library, "rb") as fp:
        embedding_library = pickle.load(fp)
    files = os.listdir(folder)
    files = [file.split('_')[0] for file in files]
    files = set(files)
    files = list(files)
    files.sort()
    files = files[-9:-1]
    # print(files)
    min_value = float('inf')
    for idx, file in enumerate(files):
        if idx +8 > len(files):
            break
        init_file  = file
        cnt = 0
        front_imgs = []
        while cnt < 8:
             file = str(int(init_file) + 2*cnt)
             front_file = file + "_0.png"
             front_img = Image.open(os.path.join(args.root_dir, folder, front_file)).convert("RGB")
             front_imgs.append(front_img)
             cnt += 1

        front_imgs = transform(front_imgs)
        recover_imgs = front_imgs.view((args.num_frames, -1) + front_imgs.size()[1:])
        for recover_img in recover_imgs:
            save_image(recover_img, "a.png")
        print(front_imgs.shape)

        front_imgs = front_imgs.cuda()
        front_imgs = front_imgs.unsqueeze(0)
        with torch.no_grad():
            embedding_feature = model(front_imgs)
        cos_distances = get_distance(embedding_feature, embedding_library)
        similarity = torch.max(cos_distances).item()
        print(similarity)
        if min_value > similarity:
            min_value = similarity
            file_name = init_file
    print(file_name, min_value)
        


if __name__ == "__main__":
    print("hello world")