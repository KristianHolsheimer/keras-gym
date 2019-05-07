#!/bin/bash
set -e

# start with fresh state
sudo apt-get dist-update -y


######################
##  NVIDIA DRIVERS  ##
######################

# cuda repo
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i ~/Downloads/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo apt-get update

# ml repo
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo dpkg -i ~/Downloads/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

# nvidia driver
sudo apt-get install --no-install-recommends -y nvidia-driver-410

# cuda
sudo apt-get install --no-install-recommends -y \
    cuda-10-0 \
    libcudnn7=7.4.1.5-1+cuda10.0  \
    libcudnn7-dev=7.4.1.5-1+cuda10.0

# nvidiaRT
sudo apt-get update && \
        sudo apt-get install -y nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
        && sudo apt-get update \
        && sudo apt-get install -y --no-install-recommends libnvinfer-dev=5.0.2-1+cuda10.0


###############
## KERAS-GYM ##
###############

sudo apt-get install -y python3-pip

mkdir -d ~/git_tree
cd ~/git_tree
git clone https://github.com/KristianHolsheimer/keras-gym.git
cd ~/git_tree/keras-gym
make install_dev
make tf_gpu
