import os
import time
import torch
import torch.nn as nn
import torch.nn.functional
import random
from einops import rearrange
from utils.preprocess import get_rel_pose, get_rel_traj
from models.stt import SpatialTemporalTransformer
from models.flux_dit import FluxParams, FluxDiT
from models.traj_dit import TrajDiT, TrajParams
from models.modules.tokenizer import poses_to_indices, yaws_to_indices
from utils.fft_utils import freq_mix, ideal_low_pass_filter
from models.modules.sampling import prepare_ids, get_schedule

class TrainTransformersDiT(nn.Module):
    def __init__(
        self,
        args,
        local_rank=-1, 
        load_path=None, 
        condition_frames=3,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.args = args
        self.condition_frames = condition_frames
        self.vae_emb_dim = self.args.vae_embed_dim * self.args.patch_size ** 2
        self.image_size = self.args.image_size
        self.traj_len = self.args.traj_len
        self.h, self.w = (self.image_size[0]//(self.args.downsample_size*self.args.patch_size),  self.image_size[1]//(self.args.downsample_size*self.args.patch_size))
        self.pkeep = args.pkeep

        self.img_token_size = self.h * self.w
        self.pose_x_vocab_size = self.args.pose_x_vocab_size
        self.pose_y_vocab_size = self.args.pose_y_vocab_size
        self.yaw_vocab_size = self.args.yaw_vocab_size
        self.pose_x_bound = self.args.pose_x_bound
        self.pose_y_bound = self.args.pose_y_bound
        self.yaw_bound = self.args.yaw_bound

        self.pose_token_size = 2 * self.args.block_size
        self.yaw_token_size = 1 * self.args.block_size
        self.traj_token_size = self.pose_token_size + self.yaw_token_size
        self.total_token_size = self.img_token_size + self.pose_token_size + self.yaw_token_size
        self.token_size_dict = {
            'img_tokens_size': self.img_token_size,
            'pose_tokens_size': self.pose_token_size,
            'yaw_token_size': self.yaw_token_size,
            'total_tokens_size': self.total_token_size
        }
        
        self.model = SpatialTemporalTransformer(
            block_size=condition_frames*(self.total_token_size),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            pose_x_vocab_size=self.pose_x_vocab_size,
            pose_y_vocab_size=self.pose_y_vocab_size,
            yaw_vocab_size=self.yaw_vocab_size,
            latent_size=(self.h, self.w), 
            # L=self.img_token_size, 
            local_rank=local_rank, 
            condition_frames=self.condition_frames, 
            token_size_dict=self.token_size_dict,
            vae_emb_dim = self.vae_emb_dim,
            temporal_block=self.args.block_size
        )
        self.model.cuda()
        
        self.dit = FluxDiT(FluxParams(
            in_channels=self.vae_emb_dim,        # origin: 64
            out_channels=self.vae_emb_dim,
            vec_in_dim=args.n_embd*(self.total_token_size-self.img_token_size),              # origin: 768
            context_in_dim=args.n_embd,          # origin: 4096
            hidden_size=args.n_embd_dit,         # origin: 3072
            mlp_ratio=4.0,
            num_heads=args.n_head_dit,           # origin: 24
            depth=args.n_layer[1],               # origin: 19
            depth_single_blocks=args.n_layer[2], # origin: 38
            axes_dim=args.axes_dim_dit,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,                # origin: True
        ))
        self.dit.cuda()
        
        self.traj_dit = TrajDiT(TrajParams(
            in_channels=self.traj_token_size,
            out_channels=self.traj_token_size,
            context_in_dim=args.n_embd,
            hidden_size=args.n_embd_dit_traj,
            mlp_ratio=4.0,
            num_heads=args.n_head_dit_traj,
            depth=args.n_layer_traj[0],
            depth_single_blocks=args.n_layer_traj[1],
            axes_dim=args.axes_dim_dit_traj,
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        ))
        self.traj_dit.cuda()
        
        bs = args.batch_size * condition_frames
        self.img_ids, self.cond_ids, self.traj_ids = prepare_ids(bs, self.h, self.w, self.total_token_size, self.traj_len)                
        self.lambda_yaw_pose = self.args.lambda_yaw_pose

        if load_path is not None:
            # load_model_path = os.path.join(load_path, 'tvar'+'_%d.pkl'%(resume_step))
            state_dict = torch.load(load_path, map_location='cpu')["model_state_dict"]
            model_state_dict = self.model.state_dict()
            for k in model_state_dict.keys():
                model_state_dict[k] = state_dict['module.model.'+k]
            self.model.load_state_dict(model_state_dict)
            traj_dit_state_dict = self.traj_dit.state_dict()
            if any(k.startswith('module.traj_dit.') for k in state_dict.keys()):
                for k in traj_dit_state_dict.keys():
                    traj_dit_state_dict[k] = state_dict['module.traj_dit.'+k]
                self.traj_dit.load_state_dict(traj_dit_state_dict)
            dit_state_dict = self.dit.state_dict()
            for k in dit_state_dict.keys():
                dit_state_dict[k] = state_dict['module.dit.'+k]
            self.dit.load_state_dict(dit_state_dict)
            print(f"Successfully load model from {load_path}")

    def normalize_traj(self, traj_targets):
        traj_targets[..., 0:1] = 2 * traj_targets[..., 0:1] / self.pose_x_bound - 1
        traj_targets[..., 1:2] /= self.pose_y_bound
        traj_targets[..., 2:3] /= self.yaw_bound
        return traj_targets
    
    def denormalize_traj(self, traj_targets):
        traj_targets[..., 0:1] = (traj_targets[..., 0:1] + 1) * self.pose_x_bound / 2
        traj_targets[..., 1:2] *= self.pose_y_bound
        traj_targets[..., 2:3] *= self.yaw_bound
        return traj_targets
        
    def model_forward(self, feature_total, rot_matrix, targets, rel_pose_cond=None, rel_yaw_cond=None, step=0):
        if (rel_pose_cond is not None) and (rel_yaw_cond is not None):
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_gt, rel_yaw_gt = get_rel_pose(rot_matrix[:, (self.condition_frames-1)*self.args.block_size:(self.condition_frames+1)*self.args.block_size])
            rel_pose_total = torch.cat([rel_pose_cond, rel_pose_gt[:, -1:]], dim=1)
            rel_yaw_total = torch.cat([rel_yaw_cond, rel_yaw_gt[:, -1:]], dim=1)
        else:
            with torch.cuda.amp.autocast(enabled=False):
                rel_pose_total, rel_yaw_total = get_rel_pose(rot_matrix[:, :(self.condition_frames+1)*self.args.block_size])

        pose_indices_total = poses_to_indices(rel_pose_total, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, t+n, 2)
        yaw_indices_total = yaws_to_indices(rel_yaw_total, self.yaw_vocab_size)  # (b, t+n, 1)
        logits = self.model(feature_total, pose_indices_total, yaw_indices_total, drop_feature=self.args.drop_feature) # 输入 b F L c 进去
        stt_features = logits['logits']
        pose_emb = logits['pose_emb']
        
        with torch.cuda.amp.autocast(enabled=False):
            traj_poses, traj_yaws = get_rel_traj(rot_matrix, self.condition_frames, self.traj_len)
        traj_targets = torch.cat([traj_poses, traj_yaws], dim=-1)   # (B, F, N, 3)
        traj_targets = traj_targets.reshape(-1, *traj_targets.shape[2:])
        traj_targets = self.normalize_traj(traj_targets)
        traj_targets = traj_targets.to(dtype=torch.bfloat16)
        yaw_pose_loss_terms = self.traj_dit.training_losses(
                        traj=traj_targets,
                        traj_ids=self.traj_ids,
                        cond=stt_features,
                        cond_ids=self.cond_ids,
                        t=torch.rand((traj_targets.shape[0], 1, 1), device=traj_targets.device),
                        return_predict=self.args.return_predict_traj
                    )
        yaw_pose_loss = self.lambda_yaw_pose * yaw_pose_loss_terms['loss']
        traj_predict = yaw_pose_loss_terms['predict']
        
        loss_terms = self.dit.training_losses(
                        img=targets,
                        img_ids=self.img_ids,
                        cond=stt_features,
                        cond_ids=self.cond_ids,
                        t=torch.rand((targets.shape[0], 1, 1), device=targets.device),
                        y=pose_emb,
                        return_predict=self.args.return_predict
                    )
        diff_loss = loss_terms['loss']
        predict = loss_terms['predict']
                    
        loss_all = diff_loss + yaw_pose_loss
        loss = {
            "loss_all": loss_all,
            "loss_diff": diff_loss,
            "loss_yaw_pose": yaw_pose_loss,
            "predict": None if not self.args.return_predict else predict,
            "predict_traj": None if not self.args.return_predict_traj else traj_predict,
        }
        return loss

    def step_train(self, latents, rot_matrix, latents_gt, rel_pose_cond=None, rel_yaw_cond=None, latents_aug=None, step=0):
        self.model.train()
                
        if latents_aug is None:
            latents_total = torch.cat([latents, latents_gt], dim=1)
        else:
            latents_total = latents_aug

        pro = random.random()
        if  pro < self.args.mask_data:
            mask = torch.bernoulli(random.uniform(0.7, 1) * torch.ones_like(latents_total))
            mask = mask.round().to(dtype=torch.int64)
            noise = torch.randn_like(latents_total)
            
            if random.random() < 0.5:
                LPF = ideal_low_pass_filter(latents_total.shape, d_s=random.uniform(0.5, 1), dims=(-1,)).cuda()
                latents_total = freq_mix(latents_total, noise, LPF, dims=(-1,))
            else:
                latents_total = mask * latents_total + (1 - mask) * noise
                
        targets = torch.cat([latents, latents_gt], dim=1)[:, 1:]
        targets = rearrange(targets, 'B F L C -> (B F) L C')
        loss = self.model_forward(latents_total, rot_matrix, targets, rel_pose_cond=rel_pose_cond, rel_yaw_cond=rel_yaw_cond, step=step)
        return loss

    def forward(self, latents, rot_matrix, latents_gt, rel_pose_cond=None, rel_yaw_cond=None, latents_aug=None, sample_last=True, step=0, **kwargs):
        if self.training:
            return self.step_train(latents, rot_matrix, latents_gt, rel_pose_cond, rel_yaw_cond, latents_aug, step)
        else:
            return self.step_eval(latents, rot_matrix, sample_last=sample_last, **kwargs)
    
    @torch.no_grad()
    def step_eval(self, latents, rel_pose, rel_yaw, sample_last=True, self_pred_traj=True, traj_only=False):
        self.model.eval()
        start_time = time.time()
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, t+1, 2)
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size)  # (b, t+1, 1)
        
        stt_features, pose_emb = self.model.evaluate(latents, pose_total, yaw_total, sample_last=sample_last) # (b F) L c
        interval = time.time() - start_time
        print("MST time:{:.2f}", interval)
        bsz = stt_features.shape[0]
        img_ids, cond_ids, traj_ids = prepare_ids(bsz, self.h, self.w, self.total_token_size, self.traj_len)
        
        start_time = time.time()
        self.traj_dit.eval()
        noise_traj = torch.randn(bsz, self.traj_len, self.traj_token_size).to(stt_features)
        timesteps_traj = get_schedule(int(self.args.num_sampling_steps), self.traj_len)
        predict_traj = self.traj_dit.sample(noise_traj, traj_ids, stt_features, cond_ids, timesteps_traj)
        predict_traj = self.denormalize_traj(predict_traj)
        interval = time.time() - start_time
        print("TrajDiT time:{:.2f}", interval)
        
        if traj_only:
            predict_latents = None
        else:
            if self_pred_traj:
                predict_pose, predict_yaw = predict_traj[:, 0:1, 0:2], predict_traj[:, 0:1, 2:3]
                predict_pose = poses_to_indices(predict_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, 1, 2)
                predict_yaw = yaws_to_indices(predict_yaw, self.yaw_vocab_size)  # (b, 1, 1)
                pose_emb = self.model.get_pose_emb(predict_pose, predict_yaw)
            
            start_time = time.time()
            self.dit.eval()
            noise = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(stt_features)
            timesteps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
            predict_latents = self.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
            predict_latents = rearrange(predict_latents, 'b (h w) c -> b h w c', h=self.h, w=self.w)
            interval = time.time() - start_time
            print("VisDiT time:{:.2f}", interval)
            
        return predict_traj, predict_latents

    @torch.no_grad()
    def generate_gt_pose_gt_yaw(self, latents, rel_pose, rel_yaw, sample_last=True):
        self.model.eval()
        pose_total = poses_to_indices(rel_pose, self.pose_x_vocab_size, self.pose_y_vocab_size)  # (b, t+1, 2)
        yaw_total = yaws_to_indices(rel_yaw, self.yaw_vocab_size)  # (b, t+1, 1)
        
        stt_features, pose_emb = self.model.evaluate(latents, pose_total[:, :(self.condition_frames+1)*self.args.block_size], yaw_total[:, :(self.condition_frames+1)*self.args.block_size], sample_last=sample_last) # (b F) L c

        self.dit.eval()
        bsz = stt_features.shape[0]
        noise = torch.randn(bsz, self.img_token_size, self.vae_emb_dim).to(stt_features)
        timesteps = get_schedule(int(self.args.num_sampling_steps), self.img_token_size)
        img_ids, cond_ids, traj_ids = prepare_ids(bsz, self.h, self.w, self.total_token_size, self.traj_len)
        predict_latents = self.dit.sample(noise, img_ids, stt_features, cond_ids, pose_emb, timesteps)
        predict_latents = rearrange(predict_latents, 'b (h w) c -> b h w c', h=self.h, w=self.w)
        return predict_latents
    
    def save_model(self, path, epoch, rank=0):
        if rank == 0:
            torch.save(self.model.state_dict(),'{}/tvar_{}.pkl'.format(path, str(epoch)))  
