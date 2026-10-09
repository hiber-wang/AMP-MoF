import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .diffusion_utils import mean_flat
from matplotlib import pyplot as plt

class RectifiedFlow:
    def __init__(
        self,
        *,
        pos_emb_scale=1000,
        num_sampling_steps=20,
        num_frames=14,
        h=12, w=24, patch=1,
    ):
        self.pos_emb_scale = pos_emb_scale
        self.num_sampling_steps = int(num_sampling_steps)
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.p = patch
        
    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, dynamics_enhance=False, return_predict=False, vis_root_path=""):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        terms = {}
        
        x_t = t * x_start + (1. - t) * noise
        target = x_start - noise
        model_output = model(x_t, t.squeeze(-1) * self.pos_emb_scale, **model_kwargs)
        assert model_output.shape == target.shape == x_start.shape
        predict = x_t + model_output * (1. - t)
        terms["mse"] = mean_flat((target - model_output) ** 2)
        if dynamics_enhance:
            predict_seq = rearrange(predict, "(b t h w) (c p q) -> b t (h p) (w q) c", t=self.num_frames, h=self.h, w=self.w, p=self.p, q=self.p)
            x_start_seq = rearrange(x_start, "(b t h w) (c p q) -> b t (h p) (w q) c", t=self.num_frames, h=self.h, w=self.w, p=self.p, q=self.p)
            bs = x_start_seq.shape[0]
            aux_loss = ((x_start_seq[:, 1:] - x_start_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])) ** 2
            
            if vis_root_path:
                aux_loss_clone = aux_loss.clone()
                aux_loss_h, aux_loss_w, aux_loss_c = aux_loss_clone.shape[-3:]
                # aux_loss_clone = rearrange(aux_loss_clone, "b t (h w) c -> b t h w c", h=aux_loss_h, w=aux_loss_w)
                aux_loss_ = F.normalize(aux_loss_clone[:, :4].reshape(bs, -1), p=1).reshape(bs, 4, aux_loss_c, aux_loss_h, aux_loss_w)
                
                predict_vis = rearrange(predict_seq, "b t h w c -> b t c h w")[0, 1:5]
                target_vis = rearrange(x_start_seq, "b t h w c -> b t c h w")[0, 1:5]
                standard_loss = (predict_vis - target_vis) ** 2
                standard_loss_ = (standard_loss * aux_loss_[0]).sum(dim=1).detach().cpu().numpy()
                standard_loss = standard_loss.sum(dim=1).detach().cpu().numpy()
                aux_loss_ = aux_loss_.sum(dim=2)
                loss_map = aux_loss_[0].detach().cpu().numpy()
                loss_map_min = loss_map.min()
                loss_map_max = loss_map.max()
                loss_map = (loss_map - loss_map_min) / (loss_map_max - loss_map_min)
                for map_id in range(4):
                    fig, ax = plt.subplots()
                    ax.matshow(loss_map[map_id], cmap="viridis", interpolation="lanczos", vmin=0, vmax=1)
                    file_name = os.path.join(vis_root_path, f"weight_viridis_first_{map_id}.png")
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
                    plt.close()
                    fig, ax = plt.subplots()
                    ax.matshow(loss_map[map_id], cmap="hot", interpolation="lanczos", vmin=0, vmax=1)
                    file_name = os.path.join(vis_root_path, f"weight_hot_first_{map_id}.png")
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
                    plt.close()
                    fig, ax = plt.subplots()
                    ax.matshow(standard_loss[map_id], cmap="viridis", interpolation="lanczos")
                    file_name = os.path.join(vis_root_path, f"diffloss_before_viridis_first_{map_id}.png")
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
                    plt.close()
                    fig, ax = plt.subplots()
                    ax.matshow(standard_loss[map_id], cmap="hot", interpolation="lanczos")
                    file_name = os.path.join(vis_root_path, f"diffloss_before_hot_first_{map_id}.png")
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
                    plt.close()
                    fig, ax = plt.subplots()
                    ax.matshow(standard_loss_[map_id], cmap="viridis", interpolation="lanczos")
                    file_name = os.path.join(vis_root_path, f"diffloss_after_viridis_first_{map_id}.png")
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
                    plt.close()
                    fig, ax = plt.subplots()
                    ax.matshow(standard_loss_[map_id], cmap="hot", interpolation="lanczos")
                    file_name = os.path.join(vis_root_path, f"diffloss_after_hot_first_{map_id}.png")
                    plt.axis("off")
                    plt.gca().set_axis_off()
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
                    plt.close()

            aux_loss = rearrange(aux_loss, "b t h w c -> b (t h w) c")
            aux_w = F.normalize(aux_loss, p=2)
            aux_w = rearrange(aux_w, "b (t h p w q) c -> b t h w (c p q)", t=self.num_frames-1, h=self.h, w=self.w, p=self.p, q=self.p)
            aux_w = 1 + torch.cat((torch.zeros(aux_w.shape[0], 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
            aux_w = rearrange(aux_w, "b t h w c -> (b t h w) c")
            terms["loss"] = mean_flat((target - model_output) ** 2 * aux_w.detach())
        else:
            terms["loss"] = terms["mse"]
        if return_predict:
            terms["predict"] = predict
        return terms

    def sample(
        self,
        model,
        shape,
        noise=None,
        model_kwargs=None
    ):
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            z = noise
        else:
            z = torch.randn(*shape).cuda()
        for i in range(self.num_sampling_steps):
            t = torch.ones((shape[0])).to(z.device) * i / self.num_sampling_steps
            vel_pred = model(z, t * self.pos_emb_scale, **model_kwargs)
            z = z.detach().clone() + vel_pred * (1 / self.num_sampling_steps)
        return z
    
    # def training_losses_(self, model, x_start, t, model_kwargs=None, noise=None, dynamics_enhance=False, return_predict=False):
    #     if model_kwargs is None:
    #         model_kwargs = {}
    #     if noise is None:
    #         noise = torch.randn_like(x_start)
    #     terms = {}
        
    #     x_t = t * x_start + (1. - t) * noise
    #     target = x_start - noise
    #     model_output = model(x_t, t.squeeze(-1) * self.pos_emb_scale, **model_kwargs)
    #     assert model_output.shape == target.shape == x_start.shape
    #     predict = x_t + model_output * (1. - t)
    #     terms["mse"] = mean_flat((target - model_output) ** 2)
    #     if dynamics_enhance:
    #         predict_seq = rearrange(predict, "(b t l) c -> b t l c", t=self.num_frames, l=self.seq_len)
    #         x_start_seq = rearrange(x_start, "(b t l) c -> b t l c", t=self.num_frames, l=self.seq_len)
    #         aux_loss = ((x_start_seq[:, 1:] - x_start_seq[:, :-1]) - (predict_seq[:, 1:] - predict_seq[:, :-1])) ** 2
    #         aux_loss = rearrange(aux_loss, "b t l c -> b (t l) c")
    #         aux_w = F.normalize(aux_loss, p=2)
    #         aux_w = rearrange(aux_w, "b (t l) c -> b t l c", t=self.num_frames-1, l=self.seq_len)
    #         aux_w = 1 + torch.cat((torch.zeros(aux_w.shape[0], 1, *aux_w.shape[2:]).to(aux_w), aux_w), dim=1)
    #         aux_w = rearrange(aux_w, "b t l c -> (b t l) c")
    #         terms["loss"] = mean_flat((target - model_output) ** 2 * aux_w.detach())
    #     else:
    #         terms["loss"] = terms["mse"]
    #     if return_predict:
    #         terms["predict"] = predict
    #     return terms