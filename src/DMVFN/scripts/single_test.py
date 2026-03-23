import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.util import *
from model.model import Model

device = torch.device("cuda")
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--image_0_path', required=True, type=str, help='image 0 path')
parser.add_argument('--image_1_path', required=True, type=str, help='image 1 path')
parser.add_argument('--load_path', required=True, type=str, help='model path')
parser.add_argument('--output_dir', default="pred.png", type=str, help='output path')

args = parser.parse_args()

def evaluate(model, args):

    with torch.no_grad():
        img_0 = cv2.imread(args.image_0_path).astype("float32")
        img_1 = cv2.imread(args.image_1_path).astype("float32")

        imgs = [img_0, img_1]
        for i in range(2):
            imgs[i] = torch.from_numpy(imgs[i].copy()).permute(2, 0, 1)
        imgs = torch.stack(imgs, 0)
        imgs = imgs.unsqueeze(0)
        imgs = imgs.to(device, non_blocking=True) / 255.
        preds = model.eval(imgs, 'single_test') # 1CHW
        print(len(preds))
        # preds = np.array(pred.cpu().squeeze() * 255).transpose(1, 2, 0) # CHW -> HWC
        # cv2.imwrite(args.output_dir, pred)
            
if __name__ == "__main__":    
    model = Model(load_path=args.load_path, training=False)
    evaluate(model, args)
        