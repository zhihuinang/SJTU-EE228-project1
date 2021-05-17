#!/bin/bash

OPT = ""
OPT+="--data_name octmnist "
OPT+="--num_epoch 10 "
OPT+="--model ResNet50 "

CUDA_VISIBLE_DEVICES=3 python3  ../main.py $OPT
