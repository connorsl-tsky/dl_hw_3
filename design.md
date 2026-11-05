# design

find . -type f -not -path './.git*' -not -path './.venv*' | xargs sed -i 's/\r//g'
removes carriage returns for crlf to lf. just in case i need it

## preprocessing
from gfg
Image from PIL
img = Image.open('.png').convert('RGB')
img = img.resize((x, y)) (if necessary)
a = np.asarray(img) / 255.0
np.expand dims to add an axis
maybe can np.reshape


np.squeeze to remove an axis
Image.fromarray() to turn array to image
img.save()
img.show()


standardization
/ 255, probably
pretty easy


data augmentation

    from unet:
    Data augmentation is essential to teach the network the desired invariance and
    robustness properties, when only few training samples are available. In case of
    microscopical images we primarily need shift and rotation invariance as well as
    robustness to deformations and gray value variations. Especially random elas-
    tic deformations of the training samples seem to be the key concept to train
    a segmentation network with very few annotated images. We generate smooth
    deformations using random displacement vectors on a coarse 3 by 3 grid. The
    displacements are sampled from a Gaussian distribution with 10 pixels standard
    deviation. Per-pixel displacements are then computed using bicubic interpola-
    tion. Drop-out layers at the end of the contracting path perform further implicit
    data augmentation.

    

keras.layers.RandomElasticTransform
keep factor at 1
keep scale at 1
maybe change fill_mode to nearest
maybe add a seed

keras.layers.RandomRotation()
maybe factor 0.5?
fill maybe nearest?
yes seed

we'll have to play with it

keras.layers.RandomShear()
maybe 0.5, 0.5

random translation?

random grayscale
keras.layers.RandomGrayscale()

how do we implement data augmentation


https://www.tensorflow.org/tutorials/images/data_augmentation
i might try this to do something different than claude

## model

we can kinda copy from gfg
at least to what makes sense to us
and edit it from there 

## submission

create a directory of all the mask images

run `python make_submission /mask_dir`


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
no mentioned optimization
good initial weights - gaussian with std of sqrt(2/N), N=number of incoming nodes of one neuron, hxwxd = 3x3x64
data augmentation - shift, rotation, defomratinos, gray value variation
random elastic deformations
drop out layers at end of contracting path

contracting block
conv 3x3, relu (-2 l/w) (unpadded)
conv 3x3, relu
max pool 2x2 (/2 l/w) (stride 2)

dropout?

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

