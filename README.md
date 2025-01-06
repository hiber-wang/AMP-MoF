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

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 \
./scripts/train.py \
--train_dataset path_of_train_dataset \
--val_datasets path_of_val_dataset \
--batch_size 8 \
--num_gpu 8
```

#### 😋Testing

```bash
python3 ./scripts/test.py \
--val_datasets path_of_test_dataset \
--load_path path_of_pretrained_weights \
```

## Recommend
We sincerely recommend some related papers:

ICSE20 - [Misbehaviour Prediction for Autonomous Driving Systems](https://github.com/testingautomated-usi/selforacle)

ASE22 - [ThirdEye: Attention Maps for Safe Autonomous Driving Systems](https://github.com/tsigalko18/ase22)
