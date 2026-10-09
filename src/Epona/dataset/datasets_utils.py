import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import math
import os
import json
from scipy.spatial.transform import Rotation
import math
from scipy.spatial.transform import Rotation as R

def reverse_seq_data(poses, seqs):
    seq_len = len(poses)

    reverse_seq = seqs[::-1]
    start_pose = poses.pop(-1)
    reverse_poses = [-pi for pi in poses]
    reverse_poses = [start_pose, ] + reverse_poses[::-1]

    return reverse_poses, reverse_seq


def data_aug_for_seq(imgs, img_h, img_w):
    seq_len = len(imgs)
    
    H, W, _ = imgs[0].shape

    # random resize
    random_resize_ratio = np.random.uniform(1.01, 1.4)
    if (img_w/img_h)*H < W:  
        resize_h = round(random_resize_ratio * img_h)
        resize_w = round(W / H * resize_h)

    else:
        resize_w = round(random_resize_ratio * img_w)
        resize_h = round(H / W * resize_w)

    # random crop
    crop_h = np.random.randint(0, resize_h-img_h-1)
    crop_w = np.random.randint(0, resize_w-img_w-1)

    for i, img in enumerate(imgs):
        img_i = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        imgs[i] = img_i[crop_h:crop_h+img_h, crop_w:crop_w+img_w, :]
    return imgs

def radians_to_degrees(radians):
    degrees = radians * (180 / math.pi)
    return degrees

def get_meta_data(poses):
    poses = np.concatenate([poses[0:1], poses], axis=0)
    rel_pose = np.linalg.inv(poses[:-1]) @ poses[1:]
    # rel_pose=  np.concatenate([rel_pose], axis=0)
    xyzs = rel_pose[:, :3, 3]
    xys = xyzs[:, :2]
    rel_yaws = radians_to_degrees(Rotation.from_matrix(rel_pose[:,:3,:3]).as_euler('zyx', degrees=False)[:,0])[:, np.newaxis]

    # rel_poses_yaws=np.concatenate([xys,rel_yaws[:,None]],axis=1)


    return {
        'rel_poses': xys,
        'rel_yaws': rel_yaws,
        # 'rel_poses_xyz': xyzs,
        # 'rel_poses_yaws':rel_poses_yaws,
    }

def seek_by_timestamp(query_time: float, timestamps: np.array, absolute_transform: np.array,  t_max_diff: float, interpolate=False):
    """Seek transform by given query_time. There are two mode supported:
        interpolate=True:
            query_time is legal within [timestamps[0], timestamps[1]]
            if query_time fall between (timestamps[i], timestamps[i+1]), then assume
            timestamps[i+1] - timestamps[i] < t_max_diff, otherwise raise RuntimeError

        interpolate=False:
            query_time is legal within (timestamps[0] - t_max_diff, timestamps[1] + t_max_diff)
            if query_time fall between (timestamps[i], timestamps[i+1]), then assume
            min(timestamps[i+1] - query_time, query_time - timestamps[i]) < t_max_diff.
            Note that in this mode, timestamps could be very sparse, allowing every query_time be very close
            or equal to one of timestamps.

    Args:
        query_time (float): must be within the loaded timestamps
        t_max_diff (float): maximum difference time allowed for any consecutive timestamps
        interpolate (bool, optional): whether to use interpolation or just simply find the transform with nearest timestamp. Defaults to False.

    Returns:
        ndarray: transform with shape=(4, 4)
    """
    assert isinstance(query_time, float), f"query_time must be float, not {type(query_time)}"
    assert isinstance(t_max_diff, float), f"t_max_diff must be float, not {type(t_max_diff)}"
    # if(len(self.relative_transform) == 0 and len(self.absolute_transform) == 0 and len(self.relative_translation) == 0):
    #     raise RuntimeError("No poses found, pleas load poses first")
    if len(absolute_transform) == 0:
        raise RuntimeError("No poses found, pleas load poses first")
    if(timestamps.shape[0] == 0):
        raise RuntimeError("No timestamps found, pleas load timestamps first")
    # if(len(self.absolute_transform) == 0):
    #     self.__relative2absolute()

    assert np.all(timestamps[1:, 0] >= timestamps[:-1, 0]), "timestamps must be sorted"

    # check if query_time match any timestamp in self.timestamps exactly
    equal_timestamp_index = np.where(np.isclose(timestamps[:, 0], query_time, rtol=1e-20, atol=1e-9))[0]
    if equal_timestamp_index.size > 0:
        return absolute_transform[equal_timestamp_index[0]]

    right_index = np.searchsorted(timestamps[:, 0], query_time, side="left")
    left_index = right_index - 1

    left_time_diff = query_time - timestamps[left_index] if left_index >= 0 else float("inf")
    right_time_diff = timestamps[right_index] - query_time if right_index < timestamps.shape[0] else float("inf")
    time_diff = min(left_time_diff, right_time_diff)[0]
    if time_diff > t_max_diff:
        raise RuntimeError(f"time_diff = {time_diff} is greater than t_max_diff {t_max_diff}")
    query_index = left_index if left_time_diff < right_time_diff else right_index
    output_transform = absolute_transform[query_index]

    return output_transform

def invT(transform):
    """inverse a transform matrix without using np.linalg.inv
    Args:
        transform (ndarray): input transform matrix with shape=(4,4)

    Returns:
        ndarray: output transform matrix with shape=(4,4)
    """
    R_Transposed = transform[:3, :3].T
    result = np.eye(4)
    result[:3, :3] = R_Transposed
    result[:3, 3] = -R_Transposed @ transform[:3, 3]
    return result

def __absolute2relative(absolute_transform):
    """
    Get relative pose from the current timestamp to the previous timestamp.
    """
    num_frames = len(absolute_transform)
    if num_frames == 0:
        raise RuntimeError("please load absolute first,\
            by using loadtxt()")
    relative_transform = [np.eye(4)]
    # relative_rotation = []
    # relative_translation = []
    for idx in range(num_frames - 1):
        # relative_transform_idx = invT(
        #     absolute_transform[idx + 1]) @ absolute_transform[idx]
        relative_transform_to_pre = invT(
            absolute_transform[idx]) @ absolute_transform[idx + 1] 
        relative_transform.append(relative_transform_to_pre)
        # relative_rotation.append(relative_transform[:3, :3])
        # relative_translation.append(relative_transform[:3, 3:])
    return relative_transform

def __loadarray_tum(array):
    assert array.shape[1] == 8
    timestamps = array[:, 0:1]
    length = array.shape[0]
    absolute_transforms = np.zeros((length, 4, 4))
    absolute_transforms[:, 3, 3] = 1
    absolute_transforms[:, :3, :3] = R.from_quat(array[:, 4:8]).as_matrix()
    absolute_transforms[:, :3, 3] = array[:, 1:4]
    absolute_transform = list(absolute_transforms)
    return timestamps, absolute_transform

def __loade2e_pose(arr):
    ts_loc = arr[:, :4]
    quaternion = R.from_euler("xyz", arr[:, 4:]).as_quat()
    loam_arr = np.concatenate([ts_loc, quaternion], axis=1)
    return loam_arr

def load_wheel_pose(wheel_pose_path, sampled_ts=None):
    """
    Load wheel poses. 
    """
    assert os.path.exists(
        wheel_pose_path
    ), f"wheel.txt not in {wheel_pose_path}"
    loam_array = np.loadtxt(wheel_pose_path)
    loam_arr = __loade2e_pose(loam_array)
    timestamps, absolute_transform = __loadarray_tum(loam_arr)

    sampled_poses = []
    for ts in sampled_ts:
        vcs_pose = seek_by_timestamp(
            ts, timestamps, absolute_transform, t_max_diff=0.121).astype("float32")
        sampled_poses.append(vcs_pose)
    sampled_poses = np.stack(sampled_poses, axis=0)
    return sampled_poses

def sample_timestamps(input_ts, random_start=False, seq_length=20, frames_downsample_ratio=2):
    sample_ts = np.array(input_ts).astype(np.float64) / 1000.0
    idx = np.arange(len(input_ts), step=frames_downsample_ratio)
    if random_start:
        start_id = np.random.randint(idx.shape[0]-seq_length-1)
    else:
        start_id = 0
        seq_length = 75 #idx.shape[0]
    downsample_ts = sample_ts[idx]
    downsample_ts_final = downsample_ts[start_id: start_id+seq_length]
    return downsample_ts_final, idx[start_id: start_id+seq_length]

def load_camera_info(camera_info_path, camera_name_list=['camera_front']):
    """
    Load camera intrinsic parameters.
    Args:
        camera_info_path: the path to the calibration file.
        camera_name_list: camera list to be loaded.
    """ 
    assert os.path.exists(camera_info_path), f"calibration not in {camera_info_path}"
    with open(camera_info_path, 'r') as f:
        calib_info = json.load(f)
    camera_info = {}
    for camera_name in camera_name_list:
        camera_k = np.array(calib_info[camera_name]['K'])
        camera_distor = np.array(calib_info[camera_name]['d'])
        camera_info[camera_name] = {'K': camera_k, 'D': camera_distor}
    return camera_info