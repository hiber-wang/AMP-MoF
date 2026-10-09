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
print(root_path)
sys.path.append(root_path)

from utils.utils import *
from utils.testing_utils import create_mp4_imgs
from dataset.dataset import TrainDataset
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.config_utils import Config
from utils.preprocess import get_rel_pose

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
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=500)
    
    
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
        os.makedirs(os.path.join(args.save_video_path, args.exp_name))
    
    with torch.no_grad():
        for i, (img, rot_matrix) in tqdm(enumerate(val_data)):
            video_save_path = os.path.join(args.save_video_path, args.exp_name, 'sliding_'+str(i+args.start_id))
            os.makedirs(video_save_path, exist_ok=True)
            model.eval()
            img = img.cuda()
            start_latents = tokenizer.encode_to_z(img[:, :condition_frames])            
            rot_matrix = rot_matrix.cuda()
            pose, yaw = get_rel_pose(rot_matrix)

            condition_imgs = []
            for j in range(condition_frames):
                img_pred = tokenizer.z_to_image(rearrange(start_latents[:, j, ...], 'b (h w) c -> b h w c', h=args.image_size[0]//(args.downsample_size*args.patch_size), w=args.image_size[1]//(args.downsample_size*args.patch_size))) 
                img_pred = (img_pred[0].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')[:,:,::-1]
                condition_imgs.append(img_pred)
                cv2.imwrite(os.path.join(video_save_path, '%d.png'%(j)), img_pred)
            
            for t1 in range(0, args.test_video_frames, args.temporal_patch_size):
                predict_latents_patch = []
                is_video = True
                for t2 in range(args.temporal_patch_size):
                    print(i, t1, t2)
                    if t1+t2+condition_frames > rot_matrix.shape[1]:
                        is_video = False
                        break
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predict_traj, predict_latents = model.step_eval(
                            start_latents,
                            pose[:, t1+t2:t1+t2+condition_frames+1, ...], 
                            yaw[:, t1+t2:t1+t2+condition_frames+1, ...], 
                            self_pred_traj=False
                        )
                    
                    predict_latents_patch.append(predict_latents)
                    predict_latents = rearrange(predict_latents, '(b f) h w c -> b f (h w) c', f=1)
                    start_latents = torch.cat((start_latents[:,1:condition_frames,...], predict_latents), dim=1)

                predict_latents_patch = torch.cat(predict_latents_patch, dim=0)
                img_preds = tokenizer.z_to_image(predict_latents_patch, is_video=is_video).cpu()
                for t2, img_pred in enumerate(img_preds):
                    img_pred_np = (img_pred.permute(1, 2, 0).numpy() * 255).astype('uint8')[:,:,::-1]                    
                    cv2.imwrite(os.path.join(video_save_path, '%d.png'%(t1+t2+condition_frames)), img_pred_np)
                    condition_imgs.append(img_pred_np)                            
            create_mp4_imgs(args, condition_imgs, video_save_path, fps=5)
                
def main(args):
    local_rank = 0
    model = TrainTransformersDiT(args, load_path=args.resume_path, local_rank=local_rank, condition_frames=args.condition_frames)
    test_dataset = TrainDataset('nuScenes', 'nuscenes_val.json', condition_frames=args.condition_frames+args.test_video_frames+args.traj_len, downsample_fps=args.downsample_fps, h=args.image_size[0], w=args.image_size[1])
    start_id, end_id = args.start_id, min(args.end_id, len(test_dataset))
    test_dataset = Subset(test_dataset, list(range(start_id, end_id))) # +list(range(100-10, 100, 2))+list(range(1000-10, 1000+5, 2)))

    print(f"Dataset length: {len(test_dataset)}, {start_id}-{end_id}")
    print(f"Condition frames: {args.condition_frames}")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    tokenizer = VAETokenizer(args, local_rank)

    test_sliding_window_img(test_dataloader, model, args, tokenizer)

if __name__ == "__main__":
    main(args)