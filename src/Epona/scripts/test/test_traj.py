import os
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
print(root_path)
sys.path.append(root_path)

from utils.utils import *
from utils.testing_utils import plot_trajectory
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
    parser.add_argument('--start_id', type=int, default=0)
    parser.add_argument('--end_id', type=int, default=100)
    
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
    save_path = os.path.join(args.save_video_path, args.exp_name)
    
    with torch.no_grad():
        for i, (img, rot_matrix) in tqdm(enumerate(val_data)):
            model.eval()
            img = img.cuda()
            start_latents = tokenizer.encode_to_z(img[:, :condition_frames])
            rot_matrix = rot_matrix.cuda()
            pose, yaw = get_rel_pose(rot_matrix)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predict_traj, _ = model.step_eval(
                    start_latents,
                    pose[:, :condition_frames+1, ...], 
                    yaw[:, :condition_frames+1, ...], 
                    self_pred_traj=False,
                    traj_only=True
                )
            predict_traj = predict_traj[0].cpu().numpy()
            filepath = os.path.join(save_path, f"{i:03d}.npy")
            np.save(filepath, predict_traj)
            gt_traj = get_rel_traj_test(rot_matrix[0, condition_frames-1:condition_frames+args.traj_len], args.traj_len)[0].cpu().numpy()
            plot_trajectory(predict_traj, gt_traj, save_path, i)
                            
def main(args):
    local_rank = 0
    model = TrainTransformersDiT(args, load_path=args.resume_path, local_rank=local_rank, condition_frames=args.condition_frames)
    test_dataset = NuPlan('nuplan-v1.1', 'nuplan_meta', split='test', condition_frames=args.condition_frames+args.traj_len, downsample_fps=args.downsample_fps, h=args.image_size[0], w=args.image_size[1])
    start_id, end_id = args.start_id, min(args.end_id, len(test_dataset))
    test_dataset = Subset(test_dataset, list(range(start_id, end_id)))

    print(f"Dataset length: {len(test_dataset)}, {start_id}-{end_id}")
    print(f"Condition frames: {args.condition_frames}")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    tokenizer = VAETokenizer(args, local_rank)

    test_sliding_window_img(test_dataloader, model, args, tokenizer)

if __name__ == "__main__":
    main(args)