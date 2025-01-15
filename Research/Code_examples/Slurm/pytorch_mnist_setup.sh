#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpu 
#SBATCH --time=02:15:00
#SBATCH --job-name=pytorch_mnist_setup
#SBATCH --output=mnist_setup.out
 
# Set up environment
uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0 # must use this version!!
uenv miniconda3-py39 # must use this version!!
conda create -n pytorch_env -c pytorch pytorch torchvision numpy -y
