#!/bin/bash

OPT = ""
OPT+="--data_name organmnist_axial "
OPT+="--num_epoch 20 "
OPT+="--model ResNet50 "

CUDA_VISIBLE_DEVICES=2 python3  ../main.py $OPT
