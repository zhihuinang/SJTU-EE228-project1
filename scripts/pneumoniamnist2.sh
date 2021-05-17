#!/bin/bash

OPT = ""
OPT+="--data_name pneumoniamnist "
OPT+="--num_epoch 20 "
OPT+="--model ResNet50 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
