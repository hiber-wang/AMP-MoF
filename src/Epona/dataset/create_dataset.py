from dataset.dataset import TrainDataset, TrainImgDataset
from torch.utils.data import ConcatDataset
from dataset.dataset_nuplan import NuPlan

def create_dataset(args, split='train'):
    data_list = args.train_data_list
    dataset_list = []
    for data_name in data_list:
        if data_name == 'nuplan':
            dataset = NuPlan(
                args.datasets_paths['nuplan_root'], 
                args.datasets_paths['nuplan_json_root'], 
                split=split, 
                condition_frames=args.condition_frames+(args.forward_iter+args.traj_len-1)*args.block_size, 
                block_size=args.block_size,
                downsample_fps=args.downsample_fps,
                h=args.image_size[0],
                w=args.image_size[1],
                no_pose=args.no_pose
            )
            print("Nuplan data length:", len(dataset))
        elif data_name == 'nuscense':
            dataset = TrainDataset(
                args.datasets_paths['nuscense_root'],
                args.datasets_paths['nuscense_train_json_path'], 
                condition_frames=args.condition_frames+(args.forward_iter+args.traj_len-1)*args.block_size, 
                downsample_fps=args.downsample_fps,
                h=args.image_size[0],
                w=args.image_size[1])
        elif data_name == 'nuscense_img':
            dataset = TrainImgDataset(
                args.datasets_paths['nuscense_root'], # args.train_nuscenes_path, 
                args.datasets_paths['nuscense_train_json_path'], 
                condition_frames=args.condition_frames, 
                downsample_fps=args.downsample_fps,
                reverse_seq=args.reverse_seq)
        dataset_list.append(dataset)
    
    data_array = ConcatDataset(dataset_list)
    return data_array, dataset_list