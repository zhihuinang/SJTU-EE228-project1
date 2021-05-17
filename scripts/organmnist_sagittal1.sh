#!/bin/bash

OPT = ""
OPT+="--data_name organmnist_sagittal "
OPT+="--num_epoch 20 "
OPT+="--model ResNet18 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
