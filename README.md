# Predicting Misbehavior of Autonomous Driving Systems via Future-Frame Motion Features

This project is the implementation of our paper **Predicting Misbehavior of Autonomous Driving Systems via Future-Frame Motion Features**, submitted to *IEEE Transactions on Vehicular Technology*.

We proposed a SOTA model AMP‑MoF for ADS misbehavior Prediction.

In recent years, it has become a hot topic in software engineering to predict Autonomous Driving System (ADS) misbehaviors. Although some methods (e.g., SelfOracle and ThirdEye) have been proposed to automatically predict ADS misbehaviors to avoid fatal accidents, there remain two challenges, namely **motion information acquisition** and **misbehavior measurement**.

In this study, we propose AMP‑MoF, a novel method to effectively predict the misbehavior of ADS via motion features. Specifically, we focus specifically on predicting two types of misbehavior in autonomous driving: **collisions and lane departures**.
First, AMP‑MoF collects images from the front‑facing camera of an autonomous vehicle as input, predicts future frames, and timely extracts dynamic motion features from the predicted frame sequence to address the motion information acquisition challenge.
Then, a transformer‑based deep semi‑supervised technique is proposed in AMP‑MoF to project the extracted features into a hypersphere space, which reflects the potential distribution of normal‑behavior features.
Finally, AMP‑MoF leverages a method based on the Gamma distribution to identify the behaviors with features projected outside the hypersphere as the misbehaviors, thus addressing the misbehavior measurement challenge.

We evaluate the effectiveness of AMP‑MoF on scenarios including bad weather and heavy‑traffic complex urban & highway driving environments.
Experimental results show that AMP‑MoF can accurately predict ADS misbehaviors, with an average _Precision_ value of 0.3172, and an average _Recall_ value of 0.9290. AMP‑MoF achieves an average _F‑score_ value of 0.7621, significantly outperforming the state‑of‑the‑art methods by 43.86 ~ 59.13 percentage points.

![Framework image](assert-new.jpg)

## Result
![Comparison image](result.png)

Our model is trained on one RTX 4080 GPU for 200 epochs, which takes about 35 hours.

## Usage
### Installation

```bash
git clone https://github.com/hiber-wang/AMP-MoF.git
cd AMP-MoF 
pip3 install -r requirements.txt
```

### Data

下载地址：[Data](https://pan.baidu.com/s/1eQ_qg2uqkJNGMEOZ3J8ulw?pwd=1234)

### Run

#### Training

```
python3 -W ignore main.py --train --pretrain --cal_center \
    --agent "$AGENT" \
    --oracle "$ORACLE" \
    --R "$R" \
    --abnormal_end "$ABNORMAL_END" \
    --num_frames "$NUM_FRAMES"
```

#### Fitting gamma distribution

```
python3 -W ignore main.py --cal_threshold \
    --agent "$AGENT" \
    --oracle "$ORACLE" \
    --R "$R" \
    --abnormal_end "$ABNORMAL_END" \
    --num_frames "$NUM_FRAMES"
```

#### Testing

> 
> **Note:** Before running the test, manually switch the threshold to match
> the desired setup:
> 
> 
> - **Self‑oracle** or **ThirdEyes**
> - False positive rate (FPR) of **0.05** or **0.01**
> 
> 
> Update the threshold value in the corresponding code before
> executing the command below.

```
python3 -W ignore main.py --test \
    --agent "$AGENT" \
    --oracle "$ORACLE" \
    --R "$R" \
    --abnormal_end "$ABNORMAL_END" \
    --num_frames "$NUM_FRAMES"
```

## Recommend

We sincerely recommend some related papers:

ICSE20 - [Misbehaviour Prediction for Autonomous Driving Systems](https://github.com/testingautomated-usi/selforacle)

ASE22 - [ThirdEye: Attention Maps for Safe Autonomous Driving Systems](https://github.com/tsigalko18/ase22)
