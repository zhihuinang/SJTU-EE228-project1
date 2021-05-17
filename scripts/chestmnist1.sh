#!/bin/bash

OPT = ""
OPT+="--data_name chestmnist "
OPT+="--num_epoch 10 "
OPT+="--model ResNet18 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
