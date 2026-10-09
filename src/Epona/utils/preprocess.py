
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))

import torch

def get_rel_poses_from_abs(pose_array):
    B, F, _ = pose_array.shape
    relative_poses = torch.zeros_like(pose_array)
    relative_poses[:, 0] = torch.tensor([0, 0, 0], dtype=pose_array.dtype, device=pose_array.device)

    origins = pose_array[:, :-1]
    currents = pose_array[:, 1:]

    thetas = -origins[:, :, 2]
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)

    # 旋转矩阵形状 [B, F, 2, 2]
    R = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).reshape(B, F-1, 2, 2)

    # 计算相对位姿
    rel_poses = currents - origins  # [B, F-1, 3]
    rel_poses[:, :, :2] = torch.einsum('bfij,bfj->bfi', R, rel_poses[:, :, :2])
    rel_poses[:, :, 2] = torch.atan2(torch.sin(rel_poses[:, :, 2]), torch.cos(rel_poses[:, :, 2]))

    relative_poses[:, 1:] = rel_poses
    return relative_poses[..., :2], relative_poses[..., 2:]


def get_rel_traj_from_abs(poses, condition_frames, traj_len):
    B, F, N = poses.shape[0], condition_frames, traj_len

    origins = poses[:, :F, None, :]  # 选取所有起始点，形状 [B, F, 1, 3]
    targets = torch.stack([poses[:, i:i+N, :] for i in range(1, F+1)], dim=1)  # [B, F, N, 3]

    thetas = -origins[..., 2]  # 旋转角度，形状 [B, F, 1]
    
    # 计算旋转矩阵 R，形状 [B, F, 2, 2]
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    R = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).reshape(B, F, 2, 2)

    # 计算相对位姿
    rel_poses = targets - origins  # [B, F, N, 3]
    rel_poses[..., :2] = torch.einsum('bfij,bfnj->bfni', R, rel_poses[..., :2])  # 旋转平移
    rel_poses[..., 2] = normalize_angle(rel_poses[..., 2])  # 角度归一化

    return rel_poses

def get_rel_traj_from_abs_test(poses, traj_len):
    N = traj_len
    F = 1
    origins = poses[:F, None, :]  #  [F, 1, 3]
    targets = poses[None, F:F+N, :]  # [F, N, 3]

    thetas = -origins[..., 2]  # 旋转角度，形状 [B, F, 1]
    
    # 计算旋转矩阵 R，形状 [B, F, 2, 2]
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    R = torch.stack([cos_t, -sin_t, sin_t, cos_t], dim=-1).reshape(F, 2, 2)

    # 计算相对位姿
    rel_poses = targets - origins  # [F, N, 3]
    rel_poses[..., :2] = torch.einsum('fij,fnj->fni', R, rel_poses[..., :2])
    rel_poses[..., 2] = normalize_angle(rel_poses[..., 2])

    return rel_poses

def normalize_angle_np(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def get_rel_poses_from_abs_np(poses):
    n = len(poses)
    relative_poses = np.zeros_like(poses)
    relative_poses[0] = np.array([0, 0, 0])
    
    origins = poses[:-1]
    currents = poses[1:]
    
    thetas = -origins[:, 2]
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    R = np.stack([cos_t, -sin_t, sin_t, cos_t], axis=-1).reshape(-1, 2, 2)
    
    rel_poses = currents - origins
    rel_poses[:, :2] = np.einsum('ijk,ik->ij', R, rel_poses[:, :2])
    rel_poses[:, 2] = normalize_angle(rel_poses[:, 2])
    
    relative_poses[1:] = rel_poses
    return relative_poses

def get_rel_traj_from_abs_np(poses, condition_frames, traj_len):
    B, F, N = poses.shape[0], condition_frames, traj_len

    origins = poses[:, :F, None, :]  # 选取所有起始点，形状 [B, F, 1, 3]
    targets = np.stack([poses[:, i:i+N, :] for i in range(F)], axis=1)  # [B, F, N, 3]

    thetas = -origins[..., 2]  # 旋转角度，形状 [B, F, 1]
    
    # 计算旋转矩阵 R，形状 [B, F, 2, 2]
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    R = np.stack([cos_t, -sin_t, sin_t, cos_t], axis=-1).reshape(B, F, 2, 2)

    # 计算相对位姿
    rel_poses = targets - origins  # [B, F, N, 3]
    rel_poses[..., :2] = np.einsum('bfij,bfnj->bfni', R, rel_poses[..., :2])  # 旋转平移
    rel_poses[..., 2] = normalize_angle(rel_poses[..., 2])  # 角度归一化

    return rel_poses


def radians_to_degrees(radians):
    return radians * 180.0 / torch.pi

def extract_yaw_from_rotation_matrices(rot_matrices):
    return torch.atan2(rot_matrices[..., 1, 0], rot_matrices[..., 0, 0])

def compute_rel_yaw(rot_matrices):
    if isinstance(rot_matrices, torch.Tensor):
        rot_matrices = rot_matrices.cpu().numpy()
    
    rot_matrices_3x3 = rot_matrices[..., :3, :3]
    reshaped_rot_matrices = rot_matrices_3x3.reshape(-1, 3, 3)

    rel_yaws = radians_to_degrees(
        R.from_matrix(reshaped_rot_matrices).as_euler('zyx', degrees=False)[:, 0]
    ).reshape(rot_matrices.shape[:-2] + (1,))

    if isinstance(rot_matrices, np.ndarray):
        return rel_yaws
    else:
        return torch.from_numpy(rel_yaws).cuda()

def get_rel_pose_np(poses):
    poses = poses.cpu().numpy()
    poses = np.concatenate([poses[:, 0:1], poses], axis=1)
    rot_matrices = np.linalg.inv(poses[:, :-1]) @ poses[:, 1:]

    rel_poses = rot_matrices[..., :2, 3]
    # print("???rel_pose", rel_poses)
    rel_yaws = compute_rel_yaw(rot_matrices)

    return torch.from_numpy(rel_poses).cuda(), torch.from_numpy(rel_yaws).cuda()

def get_rel_pose_tr(poses):
    poses = torch.tensor(poses, dtype=torch.float32) if not isinstance(poses, torch.Tensor) else poses
    poses = torch.cat([poses[:, 0:1], poses], dim=1)
    
    # 计算相对位姿
    inv_poses = torch.linalg.inv(poses[:, :-1])
    rot_matrices = torch.matmul(inv_poses, poses[:, 1:])
    
    # 提取平移部分
    rel_poses = rot_matrices[..., :2, 3]
    
    # 计算相对偏航角（yaw）
    rel_yaws = radians_to_degrees(extract_yaw_from_rotation_matrices(rot_matrices[..., :3, :3])).unsqueeze(-1)
    
    return rel_poses, rel_yaws

def get_rel_pose(poses):
    poses = torch.cat([poses[:, 0:1], poses], dim=1)
    rot_A = poses[:, :-1, :3, :3]
    rot_B = poses[:, 1:, :3, :3]
    trans_A = poses[:, :-1, :3, 3:]
    trans_B = poses[:, 1:, :3, 3:]
    rel_pose_rot = torch.linalg.inv(rot_A) @ rot_B
    rel_pose_trans = torch.linalg.inv(rot_A) @ (trans_B - trans_A)
    rel_poses = rel_pose_trans[..., :2, 0]
    rel_yaws = radians_to_degrees(extract_yaw_from_rotation_matrices(rel_pose_rot[..., :3, :3])).unsqueeze(-1)
    return rel_poses, rel_yaws

def get_rel_traj_tr(poses, condition_frames, traj_len):
    B, F, N = poses.shape[0], condition_frames, traj_len
    
    inv_poses = torch.linalg.inv(poses[:, :F, None])  # [B, F, 1, 4, 4]
    target_poses = poses[:, None, :F+N-1]  # [B, 1, F+N-1, 4, 4]
    
    rot_matrices = torch.matmul(inv_poses, target_poses)  # [B, F, F+N-1, 4, 4]
    
    # 选取每一帧后 N 帧的相对位姿
    indices = torch.arange(F).unsqueeze(1) + torch.arange(1, N+1).unsqueeze(0)  # [F, N]
    indices = torch.clamp(indices, max=F+N-2)
    
    rot_matrices = rot_matrices[:, torch.arange(F).unsqueeze(1), indices]  # [B, F, N, 4, 4]
    rel_poses = rot_matrices[..., :2, 3]  # [B, F, N, 2]
    rel_yaws = radians_to_degrees(extract_yaw_from_rotation_matrices(rot_matrices[..., :3, :3])).unsqueeze(-1)  # [B, F, N, 1]
        
    return rel_poses, rel_yaws

def get_rel_traj_np(poses, condition_frames, traj_len):
    B, F, N = poses.shape[0], condition_frames, traj_len
    poses = poses.cpu().numpy()
    inv_poses = np.linalg.inv(poses[:, :F, None])  # [B, F, 1, 4, 4]
    target_poses = poses[:, None, :F+N-1]  # [B, 1, F+N-1, 4, 4]

    rot_matrices = inv_poses @ target_poses  # [B, F, F+N-1, 4, 4]
    indices = np.arange(F).reshape(-1, 1) + np.arange(1, N+1).reshape(1, -1) 
    indices = np.clip(indices, a_min=0, a_max=F+N-2)
    rot_matrices = rot_matrices[:, np.arange(F)[:, None], indices]  # [B, F, N, 4, 4]
    rel_poses = rot_matrices[..., :2, 3]
    rel_yaws = compute_rel_yaw(rot_matrices)
    # print(f"rot_matrices:::::{rot_matrices.dtype}")
    # print(f"rel_yaws:::::{rel_yaws.dtype}")
    return torch.from_numpy(rel_poses).cuda(), torch.from_numpy(rel_yaws).cuda()

def get_rel_traj(poses, condition_frames, traj_len):
    B, F, N = poses.shape[0], condition_frames, traj_len
    rot_A = poses[:, :F, None][..., :3, :3]
    rot_B = poses[:, None, :F+N][..., :3, :3]
    trans_A = poses[:, :F, None][..., :3, 3:]
    trans_B = poses[:, None, :F+N][..., :3, 3:]
    rel_pose_rot = torch.linalg.inv(rot_A) @ rot_B
    rel_pose_trans = torch.linalg.inv(rot_A) @ (trans_B - trans_A)
    # print(rel_pose_rot.shape, rel_pose_trans.shape)
    indices = torch.arange(F).unsqueeze(1) + torch.arange(1, N+1).unsqueeze(0)  # [F, N]
    # indices = torch.clamp(indices, max=F+N-2)
    # print(indices)
    rel_pose_trans = rel_pose_trans[:, torch.arange(F).unsqueeze(1), indices]
    rel_poses = rel_pose_trans[..., :2, 0]
    rel_pose_rot = rel_pose_rot[:, torch.arange(F).unsqueeze(1), indices]
    rel_yaws = radians_to_degrees(extract_yaw_from_rotation_matrices(rel_pose_rot[..., :3, :3])).unsqueeze(-1)
    return rel_poses, rel_yaws

def get_rel_traj_test(poses, traj_len):
    N = traj_len
    F = 1
    rot_A = poses[:F, None][..., :3, :3]
    rot_B = poses[None, :F+N][..., :3, :3]
    trans_A = poses[:F, None][..., :3, 3:]
    trans_B = poses[None, :F+N][..., :3, 3:]
    rel_pose_rot = torch.linalg.inv(rot_A) @ rot_B
    rel_pose_trans = torch.linalg.inv(rot_A) @ (trans_B - trans_A)
    # print(rel_pose_rot.shape, rel_pose_trans.shape)
    indices = torch.arange(F).unsqueeze(1) + torch.arange(1, N+1).unsqueeze(0)  # [F, N]
    # indices = torch.clamp(indices, max=F+N-2)
    # print(indices)
    rel_pose_trans = rel_pose_trans[torch.arange(F).unsqueeze(1), indices]
    rel_poses = rel_pose_trans[..., :2, 0]
    rel_pose_rot = rel_pose_rot[torch.arange(F).unsqueeze(1), indices]
    rel_yaws = radians_to_degrees(extract_yaw_from_rotation_matrices(rel_pose_rot[..., :3, :3])).unsqueeze(-1)
    rel_traj = torch.cat([rel_poses, rel_yaws], dim=-1)
    return rel_traj
