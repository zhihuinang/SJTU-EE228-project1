#!/bin/bash

OPT = ""
OPT+="--data_name chestmnist "
OPT+="--num_epoch 20 "
OPT+="--model ResNet50 "

CUDA_VISIBLE_DEVICES=3 python3  ../main.py $OPT
