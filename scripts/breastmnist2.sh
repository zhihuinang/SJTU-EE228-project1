#!/bin/bash

OPT = ""
OPT+="--data_name breastmnist "
OPT+="--num_epoch 10 "
OPT+="--model ResNet50 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
