# DenseFusion-with-sphercial-convolution-encoder
Internship to be carried out under the supervision of @drmateo
6D pose estimation method developed during a master course. 
This method is based on the architecture of the DenseFusion method (https://github.com/j96w/DenseFusion) with the addition of a spherical convolution based encoder inspired by the DualPoseNet method (https://github.com/Gorilla-Lab-SCUT/DualPoseNet).
The method has been developed for the YCB-Video database with a possible extension to LineMod in the future.
The initial encoder is coded in TensorFlow, but the Densefusion architecture is coded in PyTorch.
So the new encoder does not use the same spherical convolution function (https://github.com/jonkhler/s2cnn/tree/master/s2cnn/soft) as the original encoder.  
To realise this method a study of many methods was carried out.
The methods to be studied are those present in the citation file.
These methods use visual detection, tactile detection or both.
As well as a tool to visualise the installation in the form of a video on the YCB database.

## Requirements

* Python 2.7/3.5/3.6 (If you want to use Python2.7 to run this repo, please rebuild the `lib/knn/` (with PyTorch 0.4.1).)
* [PyTorch 0.4.1](https://pytorch.org/) ([PyTroch 1.0 branch](<https://github.com/j96w/DenseFusion/tree/Pytorch-1.0>))
* PIL
* scipy
* numpy
* pyyaml
* logging
* matplotlib
* CUDA 7.5/8.0/9.0 (Required. CPU-only will lead to extreme slow training speed because of the loss calculation of the symmetry objects (pixel-wise nearest neighbour loss).)

## Datasets

This work is tested on two 6D object pose estimation datasets:

* [YCB_Video Dataset](https://rse-lab.cs.washington.edu/projects/posecnn/): Training and Testing sets follow [PoseCNN](https://arxiv.org/abs/1711.00199). The training set includes 80 training videos 0000-0047 & 0060-0091 (choosen by 7 frame as a gap in our training) and synthetic data 000000-079999. The testing set includes 2949 keyframes from 10 testing videos 0048-0059.


## Architecture :

![Untitled Diagram drawio (1)](https://user-images.githubusercontent.com/61682491/172632084-d7a1215f-6ff2-423f-9a4d-8843fe29d321.png)

## Results :
![seg2](https://user-images.githubusercontent.com/61682491/172631251-479a27e0-fd54-4a38-a0ba-7044ffe46cb3.png)
![segmentationresult](https://user-images.githubusercontent.com/61682491/172631256-03c411cf-e6d6-4add-b67b-66d887086ce3.png)
