import argparse

import torch
from torchvision.transforms import ToTensor
from datasets.txt2classlist import trans
import numpy as np


def get_arg():
    parser = argparse.ArgumentParser(description='Audio classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='pytorch-audio-classification-master',
        help="the theme's name of your task"
    )
    parser.add_argument(
        '-wp',
        type=str,
        default=r'D:\PythonCode\Pytorch-AudioClassification-master\workdir\exp-pytorch-audioclassification-master_2022_7_15_20_46\checkpoints\best_f1.pth',
        help="the checkpoint applied to predict"
    )
    parser.add_argument(
        '-fp',
        type=str,
        default=r'D:\PythonCode\Pytorch-AudioClassification-master\test(label0).npy',
        help="the audio feature file' path"
    )
    parser.add_argument(
        '-classes',
        type=list,
        default=trans(r'D:\PythonCode\Pytorch-AudioClassification-master\data\scatter\classes.txt'),
        help="classes list"
    )
    return parser.parse_args()

args = get_arg()
# ----------------------------------------------------------------------------------------------------------------------
# 加载好模型
if torch.cuda.is_available():
    print("Predict on cuda and there are/is {} gpus/gpu all.".format(torch.cuda.device_count()))
    print("Device name:{}\nCurrent device index:{}.".format(torch.cuda.get_device_name(), torch.cuda.current_device()))
else:
    print("Predict on cpu.")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Load weight from the path:{}.".format(args.wp))
model = torch.load(args.wp)
model = model.to(device)
inputs = np.load(args.fp)
inputs = ToTensor()(inputs).permute(1, 2, 0).unsqueeze(0).to(device)
# ----------------------------------------------------------------------------------------------------------------------
# # 前向传播进行预测
output = model(inputs)  # shape:(N*cls_n)
output_ = output.clone().detach().cpu()
_, pred = torch.max(output_, 1)  # 输出每一行(样本)的最大概率的下标
print("Prediction:", args.classes[int(pred)])








