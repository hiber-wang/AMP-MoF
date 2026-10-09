# Random seed
seed=1234

#! Dataset paths
datasets_paths=dict(
    nuscense_root='',
    nuscense_train_json_path='',
    nuscense_val_json_path='',
    
    nuplan_root= '',
    nuplan_json_root= '',
)
train_data_list=['nuscense']
val_data_list=['nuscense']

downsample_fps=10  # video clip is downsampled to * fps.
mask_data=0 #1 means all masked, 0 means all gt
image_size=(512, 1024)
pkeep=0.7 #Percentage for how much latent codes to keep.
reverse_seq=False
paug=0

# VAE configs
vae_embed_dim=32
downsample_size=32
patch_size=1
vae='DCAE_f32c32'
vae_ckpt='' #! VAE checkpoint path
add_encoder_temporal=False
add_decoder_temporal=False
temporal_patch_size=1

# World Model configs
condition_frames=10
n_layer=[12, 6, 6]
n_head=16
n_embd=2048
gpt_type='diffgpt_mar'
pose_x_vocab_size=128
pose_y_vocab_size=128
yaw_vocab_size=512

# Logs
outdir="exp/ckpt"
logdir="exp/job_log"
tdir="exp/job_tboard"
validation_dir="exp/validation"

diffusion_model_type="flow"
num_sampling_steps=100
lambda_yaw_pose=1.0

diff_only=True
forward_iter=3
multifw_perstep=10
block_size=1

n_embd_dit=2048
n_head_dit=16
axes_dim_dit=[16, 56, 56]
return_predict=True

traj_len=15
n_layer_traj=[1, 1]
n_embd_dit_traj=1024
n_head_dit_traj=8
axes_dim_dit_traj=[16, 56, 56]
return_predict_traj=True

fix_stt=False
test_video_frames=50
drop_feature=0
no_pose=False
sample_prob=[1.0]
pose_x_bound=50
pose_y_bound=10
yaw_bound=12