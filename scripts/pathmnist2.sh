#!/bin/bash

OPT = ""
OPT+="--data_name pathmnist "
OPT+="--num_epoch 10 "
OPT+="--model ResNet50 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
