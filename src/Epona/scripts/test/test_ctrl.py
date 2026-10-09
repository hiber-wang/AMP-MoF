import os
import cv2
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader, Subset

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from utils.utils import *
from utils.testing_utils import create_mp4_imgs, set_text
from dataset.dataset_nuplan import NuPlan
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.config_utils import Config
from utils.preprocess import get_rel_pose, get_rel_traj_test

def add_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video_path', type=str, default='test_videos')
    parser.add_argument('--iter', default=60000000, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--config', default='configs/dit/demo_config.py', type=str)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--resume_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--resume_step', default=0, type=int, help='continue to train, step')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--launcher', type=str, default='pytorch')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=2000)
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg

args = add_arguments()
print(args)

device = torch.device("cuda")
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

def test_sliding_window_img(val_data, model, args, tokenizer):
    condition_frames = args.condition_frames
    if not os.path.exists(os.path.join(args.save_video_path, args.exp_name)):
        os.makedirs(os.path.join(args.save_video_path, args.exp_name), exist_ok=True)
    
    with torch.no_grad():
        for i, (img, rot_matrix) in tqdm(enumerate(val_data)):
            video_save_path = os.path.join(args.save_video_path, args.exp_name, 'sliding_'+str(i))
            os.makedirs(video_save_path, exist_ok=True)
            model.eval()
            img = img.cuda()
            start_latents = tokenizer.encode_to_z(img[:, :condition_frames])            
            rot_matrix = rot_matrix.cuda()
            pose_total, yaw_total = get_rel_pose(rot_matrix)
            pose = pose_total[:, :condition_frames, :]
            yaw = yaw_total[:, :condition_frames, :]
            
            ######### Define your trajectory here #########
            '''
            1. pose_x: The relative translation in x-axis (forward is positive) between two consecutive frames (unit: meter),
                        corresponding to $\Delta x_{t-1\to t}$ in the paper.
            2. pose_y: The relative translation in y-axis (right is positive) between two consecutive frames (unit: meter),
                        corresponding to $\Delta y_{t-1\to t}$ in the paper.
            3. yaw: The relative rotation (turning left is positive) between two consecutive frames (unit: degree),
                        corresponding to $\Delta \theta_{t-1\to t}$ in the paper.
            '''
            def linspace(start, stop, num):
                if num <= 0:
                    return []
                if num == 1:
                    return [start]
                step = (stop - start) / (num - 1)
                return [start + i * step for i in range(num)]

            pose_x_list = [1]*60 + linspace(1, 0, 10) + [0]*5 + linspace(0, 2.5, 10) + linspace(2.5, 1, 10) + [1]*25
            yaw_list = [1]*10 + [-1]*10 + [0]*10 + [-1]*10 + [1]*10 + [0]*45 + [3]*20 + [0]*5
            ###############################################
            
            condition_imgs = []
            for j in range(condition_frames):
                img_pred = tokenizer.z_to_image(rearrange(start_latents[:, j, ...], 'b (h w) c -> b h w c', h=args.image_size[0]//(args.downsample_size*args.patch_size), w=args.image_size[1]//(args.downsample_size*args.patch_size))) 
                img_pred = (img_pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:,:,::-1]
                condition_imgs.append(img_pred)
                cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), img_pred)

            for t1 in range(len(yaw_list)): # range(min(pose.shape[1]-(condition_frames+1), 60)): 
                yaw_new = np.array([float(yaw_list[t1])])
                yaw_new = torch.from_numpy(yaw_new).cuda().unsqueeze(0).unsqueeze(1)
                yaw = torch.cat([yaw, yaw_new], dim=1)
                pose_new = np.array([float(pose_x_list[t1]), 0]) # you can change the conditional pose here.
                pose_new = torch.from_numpy(pose_new).cuda().unsqueeze(dim=0).unsqueeze(dim=1)
                pose = torch.cat([pose, pose_new], dim=1)
               
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predict_latents = model.generate_gt_pose_gt_yaw(
                        start_latents,
                        pose[:, t1:t1+condition_frames+1, ...], 
                        yaw[:, t1:t1+condition_frames+1, ...],
                    )
                
                predict_latents = rearrange(predict_latents, '(b F) h w c -> b F h w c', F=1)[:, -1:, ...]
                img_pred = tokenizer.z_to_image(predict_latents[:, 0, ...]) 
                img_pred_np = (img_pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:,:,::-1]
                format_fn = np.vectorize(lambda x: "{:.2f}".format(x))
                img_pred_np = set_text(img_pred_np,
                                str(format_fn(pose[0, t1+condition_frames, 0].cpu().numpy()))+", "+str(format_fn(pose[0, t1+condition_frames, 1].cpu().numpy()))+", "+str(yaw[0, t1+condition_frames, 0].cpu().numpy()))
                
                cv2.imwrite(os.path.join(video_save_path, '%d.png'%(t1+condition_frames)), img_pred_np)
                condition_imgs.append(img_pred_np)
                predict_latents = rearrange(predict_latents, 'b 1 h w c -> b 1 (h w) c')
                start_latents = torch.cat((start_latents[:,1:condition_frames,...], predict_latents), dim=1)
                
            create_mp4_imgs(args, condition_imgs, video_save_path, fps=5)
                
def main(args):
    local_rank = 0
    model = TrainTransformersDiT(args, load_path=args.resume_path, local_rank=local_rank, condition_frames=args.condition_frames)   
    val_data = NuPlan('nuplan-v1.1', 'nuplan_meta', split='test', condition_frames=57, downsample_fps=args.downsample_fps, h=args.image_size[0], w=args.image_size[1])
    test_dataset = Subset(val_data, [4096-8]) # + list(range(0, 10, 2)) + list(range(100-10, 100, 2))) # +list(range(100-10, 100, 2))+list(range(1000-10, 1000+5, 2)))

    print(f"Dataset length: {len(test_dataset)}")
    print(f"Condition frames: {args.condition_frames}")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    tokenizer = VAETokenizer(args, local_rank)

    test_sliding_window_img(test_dataloader, model, args, tokenizer)

if __name__ == "__main__":
    main(args)
    