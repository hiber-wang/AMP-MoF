#!/bin/bash
agent="vgg16"
oracle="ThirdEye"
# arr=(0.5 0.75 1 1.25 1.5)
# arr=(20 10 5)
# 过去的n帧图片
arr=(20)
R=1
abnormal_end=5
# 累计预测的n帧图片
num_frames=8
for i in ${arr[@]};
do
        # test_dir="dataset/$agent/test_$i"
        # vector_list="result/${agent}_vector_list_$i.pkl"
        abnormal_end=$i
        # R=$i
        #command="python3 -W ignore main.py --model trained_params/$agent.pth --test_dir $test_dir --pass-dir dataset/$agent/pass --failure-dir dataset/$agent/failure --algorithm vector --evaluate apfd --vector_list $vector_list --vector_flag True"
        # -----------训练代码---------------------
        # command="python3 -W ignore main.py --train --pretrain --cal_center  --agent $agent --oracle $oracle --R $R --abnormal_end $abnormal_end --num_frames $num_frames"
        # -----------拟合Gamma分布代码-----------
        # command="python3 -W ignore main.py --cal_threashold   --agent $agent --oracle $oracle --R $R --abnormal_end $abnormal_end --num_frames $num_frames"
        # -----------测试代码---------------------------------------
        command="python3 -W ignore main.py --test   --agent $agent --oracle $oracle --R $R --abnormal_end $abnormal_end --num_frames $num_frames"
        echo "Running: $agent, Oracle: $oracle, Before: $abnormal_end"
        $command
done
wait