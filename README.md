# Misbehavior Prediction of Autonomous Driving System via Motion Features

This project is the implementation of our Paper: [Misbehavior Prediction of Autonomous Driving System via Motion Features), which is submitted to **IEEE Transaction of Software Engineerring.** We proposed a SOTA model for ADS misbehavior Prediciton.

![Comparison image](./images/comparison.jpg)

supplement and correction of published paper: 1. In Figure 2(a), $I_{t-1}$ and $I_t$ should be swapped; 2. "Our model is trained on four 2080Ti GPUs for 300 epochs, which takes about 35 hours." This sentence refers to the Cityscapes training set; 3. In Table 1, GFLOPs computing for KITTI is on $256\times832$ and on $1024\times512$ for Cityscapes using [fvcore](https://github.com/facebookresearch/fvcore).
## Usage
### Installation

```bash
git clone https://github.com/hiber-wang/AMP-MoF.git
cd AMP-MoF 
pip3 install -r requirements.txt
```

### Run

#### 😆Training

For Cityscapes Dataset:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_port=4321 ./scripts/train.py \
--train_dataset CityTrainDataset \
--val_datasets CityValDataset \
--batch_size 8 \
--num_gpu 8
```

For KITTI Dataset:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_port=4321 ./scripts/train.py \
--train_dataset KittiTrainDataset \
--val_datasets KittiValDataset \
--batch_size 8 \
--num_gpu 8
```


For DAVIS and Vimeo Dataset:

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 \
--master_port=4321 ./scripts/train.py \
--train_dataset UCF101TrainDataset \
--val_datasets DavisValDataset VimeoValDataset \
--batch_size 8 \
--num_gpu 8
```

#### 🤔️Testing

**<span id="directly_download_test_splits"> Directly download test splits of different datasets</span>**

Download Cityscapes_test directly from [Google Drive](https://drive.google.com/file/d/1m5lfwGa6ugavZW9-UFrXNpQ0BWT7qn7X/view?usp=share_link).([百度网盘](https://pan.baidu.com/s/1KFxPC-zFi9LJwwed0PxEow) password: wk7k)

Download KITTI_test directly from [Google Drive](https://drive.google.com/file/d/1_J5QxvozXiLoF3xca0uh9a6dEMVZJtWs/view?usp=drive_link).([百度网盘](https://pan.baidu.com/s/1pG31uHts3lV2ieouZZovcQ) password: e7da)

Download DAVIS_test directly from [Google Drive](https://drive.google.com/file/d/10w1ox4ADtPdmBmYFhxHycFHcQg97PsK-/view?usp=drive_link).([百度网盘](https://pan.baidu.com/s/1ZydU6z5Y9DRQ1lBGQk8ynQ) password: mczk)

Download Vimeo_test directly from [Google Drive](https://drive.google.com/file/d/1ERswpm1E_eeS10XnGv9qH75k3-y96VwU/view?usp=drive_link).([百度网盘](https://pan.baidu.com/s/1fsTQBhQHfrPMVhtSf-77TA) password: 0mjo)

Run the following command to generate test results of DMVFN model. The `--val_datasets` can be `CityValDataset`, `KittiValDataset`, `DavisValDataset`, and `VimeoValDataset`. `--save_image` can be disabled.

```bash
python3 ./scripts/test.py \
--val_datasets CityValDataset [optional: KittiValDataset, DavisValDataset, VimeoValDataset] \
--load_path path_of_pretrained_weights \
--save_image 
```

**Image results**

We provide the image results of DMVFN on various datasets (Cityscapes, KITTI, DAVIS and Vimeo) in [百度网盘](https://pan.baidu.com/s/19BWu33raS49Wamw5iC96rA) (password: k7eb).

We also provide the results of DMVFN (without routing) in [百度网盘](https://pan.baidu.com/s/1pW61ITp5MFLvyQCr44SkEg) (password: 8zo9).

**Test the image results**

Run the following command to directly test the image results.

```bash
python3 ./scripts/test_ssim_lpips.py
```

#### 😋Single test

We provide a simple code to predict a `t+1` image with `t-1` and `t` images. Please run the following command:

```bash
python3 ./scripts/single_test.py \
--image_0_path ./images/sample_img_0.png \
--image_1_path ./images/sample_img_1.png \
--load_path path_of_pretrained_weights \
--output_dir pred.png
```

## Recommend
We sincerely recommend some related papers:

ECCV22 - [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://github.com/megvii-research/ECCV2022-RIFE)

CVPR22 - [Optimizing Video Prediction via Video Frame Interpolation](https://github.com/YueWuHKUST/CVPR2022-Optimizing-Video-Prediction-via-Video-Frame-Interpolation)
