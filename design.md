# design

## preprocessing

standardization

data augmentation

## model

## submission


# research


## gfg
https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/ - has code for each block, and how to convert and processes images and stuff, yeah?



## u-net
https://arxiv.org/pdf/1505.04597

U-Net: Convolutional Networks for Biomedical
Image Segmentation

Olaf Ronneberger, Philipp Fischer, and Thomas Brox

2015

yes use data augmentation
we need to learn how to do it
high momentum - 0.99
softmax of final feature map
cross entropy loss
good initial weights - gaussian with std of sqrt(2/N), N=number of incoming nodes of one neuron, hxwxd = 3x3x64
data augmentation - shift, rotation, defomratinos, gray value variation
random elastic deformations
drop out layers at end of contracting path

contracting block
conv 3x3, relu (-2 l/w) (unpadded)
conv 3x3, relu
max pool 2x2 (/2 l/w) (stride 2)

bottleneck
conv 3x3, relu (-2 ")
conv 3x3, relu

expanding block
up-conv 2x2 (x2 l/w) + concat with cropped feature map from contracting path
conv 3x3, relu (-2 ")
conv 3x3, relu

last
conv 1x1 to number of classes

all 2x2 max pooling need to have an even x/y size

