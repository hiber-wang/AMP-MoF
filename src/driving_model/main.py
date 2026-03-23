import matplotlib
# matplotlib.use('Agg')
from model_car import Resnet101Steer, Resnet101Speed, Vgg16Speed, Vgg16Steer, EpochSpeed, EpochSteer
from dataset_car import DrivingCaptures
import torch.optim as optim
import torch.nn as nn
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
import csv
from os import path
import numpy as np 
import pandas as pd 
import time
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import argparse
import cv2
from torch.optim import lr_scheduler
import os

parser = argparse.ArgumentParser(description="Training models")
parser.add_argument('--model_name', action='store', type=str, required=True)
parser.add_argument('--root_dir', type=str, default="/home/weizi/workspace/misbehavior_prediction/src/output/behavior")
parser.add_argument("--labels", type=str, default="/home/weizi/workspace/misbehavior_prediction/src/output/behavior/record.txt")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs',  type=int, default=200)
parser.add_argument('--re_train', type=int, default=0)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--col', type=str, default='speed')
args = parser.parse_args()

batch_size = args.batch_size
epochs = args.epochs

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
model_name = args.model_name
col = args.col
if model_name == 'resnet101':
    if col == 'steer':
        model = Resnet101Steer(pretrained=True)
    elif col == 'speed':
        model = Resnet101Speed(pretrained=True)

if model_name == 'vgg16':
    if col == 'steer':
        model = Vgg16Steer()
    if col == 'speed':
        model = Vgg16Speed()

if model_name == 'epoch':
    if col == 'steer':
        model = EpochSteer()
    if col == 'speed':
        model = EpochSpeed()

if args.re_train == 1:
    model.load_state_dict(torch.load( 'model/' + model_name + '/' + col + '.pt'))

model.to(device)

train_dataset = DrivingCaptures(root_dir=args.root_dir, labels=args.labels, col=col, is_train=True)
val_dataset = DrivingCaptures(root_dir=args.root_dir, labels=args.labels, col=col, is_train=False)

batch_size = args.batch_size
epochs = args.epochs
train_generator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
val_generator = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, drop_last=True)

criterion = nn.L1Loss()
# criterion = nn.MSELoss()
params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.Adam(params_to_update, lr=args.lr)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

min_loss = float('inf')
if args.re_train == 1:
    print("Retrain the", args.col, "model")
    val_loss = 0
    with torch.no_grad():
        for i, sample_batched in enumerate(val_generator):
            batch_image = sample_batched[0]
            single_batch_size = batch_image.shape[0]
            batch_y = sample_batched[1].view(single_batch_size, -1)
            batch_image = batch_image.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_image = batch_image.to(device)

            batch_y = batch_y.to(device)
            outputs = model(batch_image)
            loss = criterion(outputs, batch_y)
            running_loss = loss.item()
            val_loss += running_loss
    min_los = val_loss / i
    print("Current Loss:", min_los)
print("Start training", args.col, "model...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, sample_batched in enumerate(train_generator):
        batch_image = sample_batched[0]
        single_batch_size = batch_image.shape[0]
        batch_y = sample_batched[1].view(single_batch_size, -1)
        batch_image = batch_image.type(torch.FloatTensor)
        batch_y = batch_y.type(torch.FloatTensor)
        batch_image = batch_image.to(device)

        batch_y = batch_y.to(device)
        outputs = model(batch_image)
        # print("+", end="")
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
        total_loss += running_loss
        # if step % 10 == 0:
        #     print("Epoch %d Step %d MSE loss: %f" % (epoch, step, running_loss))
    exp_lr_scheduler.step()
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for i, sample_batched in enumerate(val_generator):
            batch_image = sample_batched[0]
            single_batch_size = batch_image.shape[0]
            batch_y = sample_batched[1].view(single_batch_size, -1)
            batch_image = batch_image.type(torch.FloatTensor)
            batch_y = batch_y.type(torch.FloatTensor)
            batch_image = batch_image.to(device)

            batch_y = batch_y.to(device)
            outputs = model(batch_image)
            # print("-|", end="")
            loss = criterion(outputs, batch_y)
            running_loss = loss.item()
            val_loss += running_loss
    print('Epoch %d  training RMSE loss: %.4f test loss: %.4f' % (epoch,  total_loss / step, val_loss / i), end=" ")
    if val_loss / i < min_loss:
        min_loss = val_loss / i
        torch.save(model.state_dict(), 'model/' + model_name + '/' + col + '.pt')
        print("update!")
    else:
        print()