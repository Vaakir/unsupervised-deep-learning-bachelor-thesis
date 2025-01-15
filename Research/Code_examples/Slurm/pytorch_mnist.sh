#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu 
#SBATCH --time=02:15:00
#SBATCH --job-name=pytorch_mnist
#SBATCH --output=mnist_test_01.out
 
# Activate environment

uenv verbose cuda-11.4.4 cudnn-11.x-8.8.0 # must use this version!!
uenv miniconda3-py39 # must use this version!!

conda activate pytorch_env
# Run the Python script that uses the GPU
python -u pytorch_mnist.py
