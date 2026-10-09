import os
import sys
import math
import time
import torch
import random
import logging
import argparse
from einops import rearrange
import numpy as np
import torch.distributed as dist
import torch.utils
from torch.utils.data import DataLoader, Subset
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from utils.merge_dataset import MixedBatchSampler
import torch.multiprocessing as mp
from dataset.create_dataset import create_dataset
from utils.config_utils import Config
import deepspeed
from utils.deepspeed_utils import get_deepspeed_config

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_path)
sys.path.append(root_path)

from utils.utils import *
from models.model import TrainTransformersDiT
from models.modules.tokenizer import VAETokenizer
from utils.comm import _init_dist_envi
from utils.running import init_lr_schedule, save_ckpt, load_parameters, add_weight_decay, save_ckpt_deepspeed, load_from_deepspeed_ckpt
from torch.nn.parallel import DistributedDataParallel as DDP

def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=60000000, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--config', default='configs/mar/demo_config.py', type=str)
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay (default: 0.01)')
    parser.add_argument('--resume_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--resume_step', default=0, type=int, help='continue to train, step')
    parser.add_argument('--load_stt_path', default=None, type=str, help='pretrained path')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--launcher', type=str, default='pytorch')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=2000)
    parser.add_argument('--load_from_deepspeed', default=None, type=str, help='pretrained path')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(args.__dict__)
    return cfg

logger = logging.getLogger('base')

def init_logs(global_rank, args):
    print('#### Initial logs.')
    log_path = os.path.join(args.logdir, args.exp_name)
    save_model_path = os.path.join(args.outdir, args.exp_name)
    tdir_path = os.path.join(args.tdir, args.exp_name)
    validation_path = os.path.join(args.validation_dir, args.exp_name)

    if global_rank == 0:
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        if not os.path.exists(log_path):
            os.makedirs(log_path)

        if not os.path.exists(tdir_path):
                os.makedirs(tdir_path)
        setup_logger('base', log_path, 'train', level=logging.INFO, screen=True, to_file=True)
        writer = SummaryWriter(tdir_path + '/train')
        writer_val = SummaryWriter(tdir_path + '/validate')

        args.writer = writer
        args.writer_val = writer_val
    else:
        args.writer = None
        args.writer_val = None
        
    args.log_path = log_path
    args.save_model_path = save_model_path
    args.tdir_path = tdir_path
    args.validation_path = validation_path

def init_environment(args):
    _init_dist_envi(args)
    
    # set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set backends
    torch.backends.cudnn.benchmark = True

def main(args):
    init_environment(args)
    
    if not args.distributed:
        start_training(0, args)
    else:
        # distributed training
        if args.launcher == 'pytorch':
            print('pytorch launcher.')
            local_rank = int(os.environ["LOCAL_RANK"])
            start_training(local_rank, args)
        elif args.launcher == 'slurm': 
            # this is for debug
            num_gpus_per_nodes = torch.cuda.device_count()
            mp.spawn(start_training, nprocs=num_gpus_per_nodes, args=(args, ))
        else:
            raise RuntimeError(f'{args.launcher} is not supported.')
        
def start_training(local_rank, args):
    torch.cuda.set_device(local_rank)

    if 'RANK' not in os.environ:
        node_rank  = 0  # when debugging, only has a single node
        global_rank = node_rank * torch.cuda.device_count() + local_rank
        os.environ["RANK"] = str(global_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
    init_logs(int(os.environ["RANK"]), args)

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])

    print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")  
    train(local_rank, args)


def train(local_rank, args):
    print(args)
    writer = args.writer
    rank = int(os.environ['RANK'])
    save_model_path = args.save_model_path

    step = args.resume_step

    model = TrainTransformersDiT(args, local_rank=local_rank, condition_frames=args.condition_frames // args.block_size)
    stt_params, dit_params, traj_params = count_parameters(model.model), count_parameters(model.dit), count_parameters(model.traj_dit)
    print(f"Total Parameters: {format_number(stt_params + dit_params)}, SST Parameters: {format_number(stt_params)}, DiT Parameters: {format_number(dit_params)}, TrajDiT Parameters: {format_number(traj_params)}")

    model = DDP(model, device_ids=[local_rank, ], output_device=local_rank, find_unused_parameters=True)
    tokenizer = VAETokenizer(args, local_rank)
    eff_batch_size = args.batch_size * args.condition_frames // args.block_size * dist.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)
    lr = args.lr

    param_groups = add_weight_decay(model.module, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    lr_schedule = init_lr_schedule(optimizer, milstones=[1000000, 1500000, 2000000], gamma=0.5)

    skip_key = None
    if args.load_stt_path is not None:
        checkpoint = torch.load(args.load_stt_path, map_location="cpu")
        print(f"Load stt: {args.load_stt_path}")
        skip_key="causal_time_space_blocks"
        model.module = load_parameters(model.module, checkpoint)
        del checkpoint
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location="cpu")
        print(f"Load model: {args.resume_path}")
        model.module = load_parameters(model.module, checkpoint, skip_key=skip_key)
        del checkpoint
    if args.fix_stt:
        for name, param in model.module.named_parameters():
            if "causal_time_space_blocks" in name:
                param.requires_grad = False
                print(f"Frozen: {name}")

    train_dataset, train_datalist = create_dataset(args)
    if args.overfit:
        train_dataset = Subset(train_dataset, list(range(4096-100, 4096+100))+list(range(0, 200)))
    
    mix_data_sampler = MixedBatchSampler(
        src_dataset_ls=train_datalist,
        batch_size=args.batch_size, 
        rank=rank, 
        seed=args.seed, 
        num_replicas=int(os.environ["WORLD_SIZE"]), 
        drop_last=True, 
        shuffle=True, 
        prob=args.sample_prob, 
        generator=torch.Generator().manual_seed(0),
    )
    train_data = DataLoader(
        train_dataset,
        num_workers=32,
        batch_sampler=mix_data_sampler
    )
    
    # sampler = DistributedSampler(train_dataset)
    # train_data = DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size, 
    #     num_workers=32, 
    #     pin_memory=True, 
    #     drop_last=True, 
    #     sampler=sampler
    # )        
    
    print('Length of train_data', len(train_data))
    epoch = step // len(train_data) + 1
    deepspeed_cfg = get_deepspeed_config(args)
    model, optimizer, _, _ = deepspeed.initialize(
        config_params=deepspeed_cfg,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_schedule,
    )
    load_from_deepspeed_ckpt(args, model)

    torch.set_float32_matmul_precision('high')
    
    print('training...')
    torch.cuda.synchronize()
    time_stamp = time.time()
    while step < args.iter:
        # lr = adjust_learning_rate(optimizer, epoch, args)
        for i, (img, rot_matrix) in enumerate(train_data):
            # print(f'into training {i}: {img.shape}, {rot_matrix.shape}')
            model.train()
            img = img.cuda()
            latents = tokenizer.encode_to_z(img[:, :-args.traj_len+args.forward_iter])
            rot_matrix = rot_matrix.cuda()# .to(torch.bfloat16)
                        
            data_time_interval = time.time() - time_stamp
            torch.cuda.synchronize()
            time_stamp = time.time()
            cf = args.condition_frames // args.block_size
            latents_cond = latents[:, :cf, ...] # B F L(H*W) C!
            rel_pose_cond, rel_yaw_cond = None, None
            fw_iter = 1
            if step % args.multifw_perstep == 0:
                fw_iter = args.forward_iter
            for j in range(fw_iter):
                # optimizer.zero_grad()
                rot_matrix_cond = rot_matrix[:, j*args.block_size:j*args.block_size+args.condition_frames+args.traj_len, ...]
                latents_gt = latents[:, j+cf:j+cf+1, ...]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss_final = model(
                        latents_cond, 
                        rot_matrix_cond,
                        latents_gt,
                        rel_pose_cond=rel_pose_cond,
                        rel_yaw_cond=rel_yaw_cond,
                        step=step
                    )
                loss_value = loss_final["loss_all"]

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)
                model.backward(loss_value)
                model.step()
                
                # if args.return_predict and rank == 0 and step % args.eval_steps == 0:
                #     predict_latents = loss_final["predict"].detach()
                #     # validation_step_path = os.path.join(args.validation_path, 'val_'+str(step))
                #     os.makedirs(args.validation_path, exist_ok=True)
                #     gt = ((img[0, 0].permute(0, 2, 3, 1).cpu().numpy() / 2 + 0.5) * 255).astype('uint8')
                #     latents_pred = rearrange(predict_latents, 'b (h w) c -> b h w c', h=args.image_size[0]//(args.downsample_size*args.patch_size), w=args.image_size[1]//(args.downsample_size*args.patch_size))
                #     imgs_pred = tokenizer.z_to_image(latents_pred.float())
                #     pred = (imgs_pred[0].cpu().numpy() * 255).astype('uint8')
                #     imgs = np.concatenate((gt, pred), axis=2)
                #     # imgs = np.concatenate(imgs, axis=0)
                #     cv2.imwrite(os.path.join(args.validation_path, str(step)+'.jpg'), imgs[:,:,::-1])
                
                if j < fw_iter - 1:
                    if args.return_predict:
                        predict_latents = loss_final["predict"].detach()
                        predict_traj = loss_final["predict_traj"].detach()
                    else:
                        model.eval()
                        predict_traj, predict_latents = model(
                            latents_cond, 
                            rot_matrix_cond,
                            latents_gt,
                            sample_last=False
                        )
                        model.train()
                    latents_cond = rearrange(predict_latents, '(b t) l c -> b t l c', b=args.batch_size, t=args.condition_frames // args.block_size)
                    rel_traj_cond = rearrange(predict_traj, '(b t) l c -> b t l c', b=args.batch_size, t=args.condition_frames // args.block_size)[:, :, 0, :]
                    rel_pose_cond, rel_yaw_cond = rel_traj_cond[..., 0:2], rel_traj_cond[..., 2:3]
            step += 1
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            dist.barrier()
            torch.cuda.synchronize()
            if step % 100 == 1 and rank == 0:
                writer.add_scalar('learning_rate/lr', lr, step)
                writer.add_scalar('loss/loss_all', loss_final["loss_all"].to(torch.float32), step)
                writer.add_scalar('loss/loss_diff', loss_final["loss_diff"].to(torch.float32), step)
                writer.add_scalar('loss/loss_yaw_pose', loss_final["loss_yaw_pose"].to(torch.float32), step)
                writer.flush()
            if rank == 0:
                logger.info('step:{} time:{:.2f}+{:.2f} lr:{:.4e} loss_avg:{:.4e} diff_loss:{:.4e} pose_loss:{:.4e} '.format( \
                    step, data_time_interval, train_time_interval, optimizer.param_groups[0]['lr'],  loss_final["loss_all"].to(torch.float32), loss_final["loss_diff"].to(torch.float32), loss_final["loss_yaw_pose"].to(torch.float32)))
            if step % args.eval_steps == 0: # or (step == 1): 
                dist.barrier()
                torch.cuda.synchronize()
                save_ckpt_deepspeed(args, save_model_path, model, optimizer, lr_schedule, step)
                dist.barrier()
                if rank == 0:
                    save_ckpt(args, save_model_path, model.module, optimizer, lr_schedule, step)
                torch.cuda.synchronize()
                dist.barrier()
        epoch += 1
        dist.barrier()
        
if __name__ == "__main__":
    os.chdir(root_path)
    args = add_arguments()
    main(args)