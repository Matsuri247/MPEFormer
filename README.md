# Mosaic Pattern Excavation Transformer for Spectral Imaging
The code implementation of paper "Multi-stage Spatial-Spectral Fusion Network for Spectral Super-Resolution". The code will be available soon as the paper is published.

# Environment
```
Python=3.9.19
opencv-python==4.9.0.80
einops
torchvision==0.14.1
torchaudio==0.13.1
torch==1.13.1
scipy==1.13.0
h5py
hdf5storage
tqdm
torchinfo
```

# Data Preparation
You can find ARAD dataset and Chikusei dataset from ([here](https://github.com/bowenzhao-zju/PPIE-SSARN)). Make sure you place the dataset as the following form:
```
|--demosaicing_MPEFormer
    |--dataset 
        |--ARAD
            |--train
                |--ARAD_1K_0001_16.mat
                |--ARAD_1K_0002_16.mat
                ： 
                |--ARAD_1K_0900_16.mat
            |--test
                |--ARAD_1K_0901_16.mat
                |--ARAD_1K_0902_16.mat
                ： 
                |--ARAD_1K_0950_16.mat
        |--Chikusei
            |--train
                |--001_16.mat
                |--002_16.mat
                ： 
                |--090_16.mat
            |--test
                |--091_16.mat
                |--092_16.mat
                ： 
                |--100_16.mat
```
