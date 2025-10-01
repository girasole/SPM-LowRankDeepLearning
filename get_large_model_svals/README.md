## Singular Value Plotting of Large Models
This is the codebase to reproduce the low-rank singular value figures for the DLN, MLP, ViT and VGG16 networks.
We provide all model checkpoints to reproduce the figures in the following link:

The folder `SPM-Weights/Header-Weights` contains the necessary files to generate the singular values plots for DLN + MLP + ViT + VGG16.

The folder `SPM-Weights/viT-Weights` contains the necessary files to generate the singular values plots for the ViT qkv and fc weights.

## Usage Examples

`python3 compute_weights_qkv.py` # This generates all 3D plots for the QKV weights of the ViT

`python3 plot_all_svals.py` # This generates the main figure with the svals of all four networks (both 2D and 3D)

`python3 sval_2d_plotter.py` # This generates the singular value plots for the ViT network across different layers

## Training the Networks

You can also train the networks using the following:

`python3 train_save_weights.py` # For ViT and VGG on CIFAR-10

`python3 dln_main.py` # For DLN on MNIST

`python3 mlp_main.py` # For MLP on MNIST
