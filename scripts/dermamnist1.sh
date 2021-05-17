#!/bin/bash

OPT = ""
OPT+="--data_name dermamnist "
OPT+="--num_epoch 20 "
OPT+="--model ResNet18 "

CUDA_VISIBLE_DEVICES=1 python3  ../main.py $OPT
