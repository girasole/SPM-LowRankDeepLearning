# Singular Value Plotting of Large Models
This is the codebase to reproduce the low-rank singular value figures for the DLN, MLP, ViT and VGG16 networks.

# Usage Examples
We provide all model checkpoints to reproduce the figures in the following link:

The folder `SPM-Weights/Header-Weights` contains the necessary files to generate the singular values plots for DLN + MLP + ViT + VGG16.

The folder `SPM-Weights/viT-Weights` contains the necessary files to generate the singular values plots for the ViT qkv and fc weights.

`python train_cifar10.py` # vit-patchsize-4

`python train_cifar10.py  --size 48` # vit-patchsize-4-imsize-48

`python train_cifar10.py --patch 2` # vit-patchsize-2

`python train_cifar10.py --net vit_small --n_epochs 400` # vit-small

`python train_cifar10.py --net vit_timm` # train with pretrained vit

`python train_cifar10.py --net convmixer --n_epochs 400` # train with convmixer

`python train_cifar10.py --net mlpmixer --n_epochs 500 --lr 1e-3`

`python train_cifar10.py --net cait --n_epochs 200` # train with cait

`python train_cifar10.py --net swin --n_epochs 400` # train with SwinTransformers

`python train_cifar10.py --net res18` # resnet18+randaug

