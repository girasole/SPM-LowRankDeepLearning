# Dropout Effects on Singular Values

This project provides tools to study the effect of dropout on the singular values of activations in neural networks. It supports various architectures (MLP, Deep Linear, ResNet, Transformer) and datasets (MNIST, CIFAR10, synthetic data).

## Features

- Multiple neural network architectures with configurable dropout
- Activation extraction from specified layers
- SVD computation and analysis of layer activations
- Checkpointing for model states
- Meta-experiments with different dropout ratios and random seeds

## Project Structure

```
data/
    data_loader.py  # Data loading and preprocessing for various datasets
models/
    architectures.py  # Model architectures with dropout support
utils/
    training.py  # Training utilities, SVD computation, and activation tracking
main.py  # Main script to run experiments
```

## Reproducing Paper Figures

This section describes how to reproduce **Figure 3** and **Figure 10** from the paper. The following commands generate both figures.

### Experiment Commands

**Deep Linear Network:**
```bash
python main.py --dataset synthetic --model deeplinear --hidden_dims 20,20 --loss mse --meta_experiment --dropout_ratios 0.0,0.3,0.6,0.9 --num_seeds 3 --epochs 20 --experiment_name deeplinear
```

**MLP:**
```bash
python main.py --dataset synthetic --model mlp --hidden_dims 20,20 --loss mse --meta_experiment --dropout_ratios 0.0,0.3,0.6,0.9 --num_seeds 3 --epochs 20 --experiment_name mlp
```

**ResNet:**
```bash
python main.py --dataset cifar10 --model resnet --loss crossentropy --meta_experiment --dropout_ratios 0.0,0.3,0.6,0.9 --num_seeds 3 --epochs 20 --experiment_name resnet
```

### Figure 3: Singular Values of Activations Over Training Epochs

Figure 3 shows the singular values of activations of the last layer in deep architectures trained with dropout rate 60%, as a function of training epoch. Deep Linear Network (DLN) and Multi-Layer Perceptron (MLP) are trained on synthetic data with MSE loss, while ResNet is trained on CIFAR-10 dataset with cross-entropy loss.

**Output Figure Paths:**

- **Figure 3(a)**: `results/<deeplinear_experiment_name>/epoch_plots/dropout_0.6/seed_42/network.4_singular_values_normalized_plot.pdf`
- **Figure 3(b)**: `results/<mlp_experiment_name>/epoch_plots/dropout_0.6/seed_42/network.6_singular_values_normalized_plot.pdf`
- **Figure 3(c)**: `results/<resnet_experiment_name>/epoch_plots/dropout_0.6/seed_42/layer4_singular_values_normalized_plot.pdf`

### Figure 10: Singular Values by Dropout Rate

Figure 10 shows the singular values of activations of the first and last layers in deep architectures from masked training with varying dropout rates. Deep Linear Network (DLN) is trained on synthetic data with MSE loss, while ResNet is trained on CIFAR-10 dataset with cross-entropy loss.

**Output Figure Paths:**

- **Figure 10(a)**: `results/<deeplinear_experiment_name>/network.0_singular_values_by_dropout_linear.pdf`
- **Figure 10(b)**: `results/<deeplinear_experiment_name>/network.4_singular_values_by_dropout_linear.pdf`
- **Figure 10(c)**: `results/<resnet_experiment_name>/layer1_singular_values_by_dropout.pdf`
- **Figure 10(d)**: `results/<resnet_experiment_name>/layer4_singular_values_by_dropout.pdf`


## Usage

### Basic Training with Singular Value Analysis

```bash
python main.py --dataset mnist --model mlp --hidden_dims 256,256 --dropout_prob 0.2 --epochs 10 --compute_svd_interval 500
```

This will train an MLP on MNIST with dropout probability 0.2, and compute SVD of activations every 500 iterations.

### Meta Experiment for Different Dropout Ratios

```bash
python main.py --dataset mnist --model mlp --hidden_dims 256,256 --meta_experiment --dropout_ratios 0.0,0.1,0.2,0.3,0.4,0.5 --num_seeds 3 --epochs 10
```

This will run multiple experiments with different dropout ratios and seeds, and aggregate the results.

### Using Different Architectures

#### Deep Linear Network

```bash
python main.py --dataset mnist --model deeplinear --hidden_dims 256,256,256 --dropout_prob 0.2
```

#### ResNet for CIFAR-10

```bash
python main.py --dataset cifar10 --model resnet --dropout_prob 0.2
```

#### Transformer

```bash
python main.py --dataset mnist --model transformer --dropout_prob 0.2
```

### Tracking Specific Layers

To track specific layers for SVD analysis:

```bash
python main.py --dataset mnist --model mlp --track_layers linear0,linear1 --compute_svd_interval 500
```

## Command Line Arguments

### Dataset Parameters
- `--dataset`: Dataset to use ('mnist', 'cifar10', 'synthetic')
- `--data_dir`: Directory to store/load data
- `--batch_size`: Batch size
- `--num_workers`: Number of workers for data loading

### Model Parameters
- `--model`: Model architecture ('deeplinear', 'mlp', 'resnet', 'transformer')
- `--hidden_dims`: Hidden dimensions, comma-separated
- `--activation`: Activation function for MLP ('relu', 'tanh', 'sigmoid', 'leaky_relu')

### Training Parameters
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--device`: Device to use for training ('cpu', 'cuda')
- `--seed`: Random seed
- `--eval_interval`: Interval (in epochs) to evaluate model
- `--compute_svd_interval`: Interval (in iterations) to compute SVD
- `--checkpoint_interval`: Interval (in iterations) to save checkpoints
- `--save_activations_interval`: Interval (in iterations) to save activations

### Dropout Parameters
- `--dropout_prob`: Dropout probability (for single run)
- `--track_layers`: Layers to track activations from (comma-separated or "all")

### Meta-experiment Parameters
- `--meta_experiment`: Run meta experiment with multiple dropout ratios and seeds
- `--dropout_ratios`: Dropout ratios for meta experiment (comma-separated)
- `--num_seeds`: Number of seeds per dropout ratio

### Output Parameters
- `--save_dir`: Directory to save results
- `--experiment_name`: Name for the experiment

## Results

The results are saved in the specified `save_dir` directory, with the following structure:

```
results/
    experiment_name/
        hyperparameters.json         # Training hyperparameters
        training_curves.png          # Plot of training and validation metrics
        checkpoints/                 # Model checkpoints
        activations/                 # Saved activations (optional)
        svd/                         # SVD results
            layer_name/
                epoch_X_iter_Y_singular_values.npy  # Singular values
                condition_numbers.csv               # Condition numbers
```

For meta-experiments, additional aggregate plots are saved:

```
results/
    meta_experiment_name/
        meta_results.json                     # Aggregated results
        test_accuracy_vs_dropout.png          # Plot of test accuracy vs dropout ratio
        val_accuracy_curves.png               # Plot of validation accuracy curves
        dropout_X_seed_Y/                     # Individual experiment results
            ...
```