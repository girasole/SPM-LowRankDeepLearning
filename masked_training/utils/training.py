import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import matplotlib.pyplot as plt
from datetime import datetime

def get_loss_function(loss_name):
    """
    Returns the appropriate loss function based on the name.
    
    Args:
        loss_name: Name of the loss function to use
        
    Returns:
        torch.nn.Module: The loss function
    """
    if loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    elif loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'l1':
        return nn.L1Loss()
    elif loss_name == 'smoothl1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")

class Trainer:
    """
    Trainer class for handling model training and evaluation with dropout.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './results',
        experiment_name: str = None,
        track_layers: Union[List[str], str] = None,
        seed: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            criterion: Loss function
            optimizer: Optimizer, if None, SGD with lr will be used
            lr: Learning rate if optimizer is not provided
            device: Device to use for training ('cpu', 'cuda')
            save_dir: Directory to save results
            experiment_name: Name of the experiment
            track_layers: List of layer names to track activations, or 'all'
            seed: Random seed for reproducibility
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set default criterion to CrossEntropyLoss if none provided
        # self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        # Set default criterion to MSELoss if none provided
        self.criterion = criterion if criterion is not None else nn.MSELoss()

        
        # Set default optimizer to SGD if none provided
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        if experiment_name is None:
            self.experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
            
        self.experiment_dir = os.path.join(save_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Create subdirectories for checkpoints, activations, and svd
        self.checkpoint_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.activations_dir = os.path.join(self.experiment_dir, 'activations')
        self.svd_dir = os.path.join(self.experiment_dir, 'svd')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.activations_dir, exist_ok=True)
        os.makedirs(self.svd_dir, exist_ok=True)
        
        # Track which layers to save activations from
        self.track_layers = track_layers
        
        # Register hooks for tracking activations
        if hasattr(self.model, 'register_hooks') and self.track_layers:
            self.model.register_hooks(self.track_layers)
        
        # Set seeds for reproducibility
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Save hyperparameters
        self.save_hyperparameters()
        
    def save_hyperparameters(self):
        """Save training hyperparameters to file"""
        hparams = {
            'optimizer': self.optimizer.__class__.__name__,
            'learning_rate': [group['lr'] for group in self.optimizer.param_groups],
            'criterion': self.criterion.__class__.__name__,
            'device': self.device,
            'seed': self.seed,
            'track_layers': self.track_layers,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save optimizer-specific parameters
        if isinstance(self.optimizer, optim.SGD):
            hparams.update({
                'momentum': self.optimizer.defaults.get('momentum', 0.0),
                'weight_decay': self.optimizer.defaults.get('weight_decay', 0.0),
                'dampening': self.optimizer.defaults.get('dampening', 0.0),
                'nesterov': self.optimizer.defaults.get('nesterov', False)
            })
        
        # Save model-specific parameters
        if hasattr(self.model, 'model') and hasattr(self.model.model, '__class__'):
            hparams['model_type'] = self.model.model.__class__.__name__
            
            # Extract dropout related parameters if applicable
            if hasattr(self.model.model, 'dropout_prob'):
                hparams['dropout_prob'] = self.model.model.dropout_prob
                
            if hasattr(self.model.model, 'dropout_locations'):
                hparams['dropout_locations'] = self.model.model.dropout_locations
                
            if hasattr(self.model.model, 'hidden_dims'):
                hparams['hidden_dims'] = self.model.model.hidden_dims
        
        # Save hyperparameters to file
        with open(os.path.join(self.experiment_dir, 'hyperparameters.json'), 'w') as f:
            json.dump(hparams, f, indent=4)
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Handle target shape for different criteria
            if self.criterion.__class__.__name__ == 'MSELoss':
                if target.dim() == 1:
                    target = target.float().unsqueeze(1)
                if output.size() != target.size():
                    # Fix the dimension issue by creating one-hot and then squeezing
                    # to match output shape [batch_size, num_classes]
                    target = F.one_hot(target.long(), output.size(1)).float().squeeze(1)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
        
        epoch_loss = running_loss / total_samples
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self, loader: torch.utils.data.DataLoader = None) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            loader: DataLoader to use for validation, defaults to self.val_loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        if loader is None:
            if self.val_loader is None:
                return 0.0, 0.0
            loader = self.val_loader
        
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                # Handle target shape for different criteria
                if self.criterion.__class__.__name__ == 'MSELoss':
                    if target.dim() == 1:
                        target = target.float().unsqueeze(1)
                    if output.size() != target.size():
                        # target = F.one_hot(target.long(), output.size(1)).float()
                        # Fix the dimension issue by creating one-hot and then squeezing
                        # to match output shape [batch_size, num_classes]
                        target = F.one_hot(target.long(), output.size(1)).float().squeeze(1)
                
                val_loss += self.criterion(output, target).item() * data.size(0)
                
                # Calculate accuracy
                if output.shape[1] > 1:  # Classification
                    pred = output.argmax(dim=1, keepdim=True)
                    if target.dim() > 1:
                        target = target.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                else:  # Regression
                    # For regression, calculate "accuracy" as percentage of predictions
                    # within 10% of target value
                    tolerance = torch.abs(target) * 0.1
                    correct += torch.abs(output - target).le(tolerance).sum().item()
                
                total += data.size(0)
        
        # Compute average loss and accuracy
        avg_loss = val_loss / total
        accuracy = correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def save_activations(self, epoch: int, iteration: int = None):
        """
        Save activations from tracked layers.
        
        Args:
            epoch: Current epoch number
            iteration: Current iteration number (if within epoch)
        """
        if not hasattr(self.model, 'get_activations'):
            return
        
        # Get activations from model
        activations = self.model.get_activations()
        
        if not activations:
            return
        
        # Create iteration identifier
        if iteration is not None:
            identifier = f"epoch_{epoch}_iter_{iteration}"
        else:
            identifier = f"epoch_{epoch}"
        
        # Save each layer's activations
        for layer_name, activation in activations.items():
            # Ensure the activation is on CPU and convert to numpy
            activation_np = activation.cpu().numpy()
            
            # Create layer-specific directory
            layer_dir = os.path.join(self.activations_dir, layer_name)
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save activation
            np.save(os.path.join(layer_dir, f"{identifier}.npy"), activation_np)
    
    def compute_svd(self, epoch: int, iteration: int = None):
        """
        Compute SVD of activations and save singular values.
        
        Args:
            epoch: Current epoch number
            iteration: Current iteration number (if within epoch)
        """
        if not hasattr(self.model, 'get_activations'):
            return
        
        # Set model to eval mode temporarily for consistent activations
        training_mode = self.model.training
        self.model.eval()
        
        # Get a batch of data for SVD computation
        data_iter = iter(self.val_loader if self.val_loader else self.train_loader)
        data_batch, _ = next(data_iter)
        data_batch = data_batch.to(self.device)
        
        # Forward pass to get activations
        with torch.no_grad():
            _ = self.model(data_batch)
        
        # Get activations
        activations = self.model.get_activations()
        
        if not activations:
            # Restore original training mode
            if training_mode:
                self.model.train()
            return
        
        # Create iteration identifier
        if iteration is not None:
            identifier = f"epoch_{epoch}_iter_{iteration}"
        else:
            identifier = f"epoch_{epoch}"
        
        # Define which layers to process
        target_layers = ["layer1", "layer4"]
        
        # Compute SVD for each layer
        for layer_name, activation in activations.items():
            # Skip layers not in our target list - check if the layer name contains any of the target layers
            if not any(target in layer_name for target in target_layers):
                continue
                
            # Skip if activation doesn't have enough dimensions
            if activation.dim() < 2:
                continue
                
            # Reshape if needed - we need a 2D matrix for SVD
            if activation.dim() > 2:
                # For convolutional layers: [batch, channels, height, width]
                batch_size = activation.size(0)
                activation = activation.view(batch_size, -1)  # Flatten to [batch, features]
            
            # Compute SVD
            try:
                U, S, V = torch.svd(activation)
                
                # Save singular values
                singular_values = S.cpu().numpy()
                
                # Create layer-specific directory
                layer_dir = os.path.join(self.svd_dir, layer_name)
                os.makedirs(layer_dir, exist_ok=True)
                
                # Save singular values as CSV
                sv_filename = os.path.join(layer_dir, f"{identifier}_singular_values.csv")
                with open(sv_filename, 'w') as f:
                    f.write("index,value\n")
                    for idx, value in enumerate(singular_values):
                        f.write(f"{idx},{value}\n")
                
                # Compute and save condition number (ratio of largest to smallest singular value)
                if len(singular_values) > 1 and singular_values[-1] > 1e-10:
                    condition_number = singular_values[0] / singular_values[-1]
                    
                    # Append to condition number log
                    with open(os.path.join(layer_dir, "condition_numbers.csv"), "a") as f:
                        if os.path.getsize(os.path.join(layer_dir, "condition_numbers.csv")) == 0:
                            f.write("epoch,iteration,condition_number\n")
                        f.write(f"{epoch},{iteration if iteration else 'NA'},{condition_number}\n")
                
            except Exception as e:
                print(f"Error computing SVD for layer {layer_name}: {e}")
        
        # Restore original training mode
        if training_mode:
            self.model.train()
    
    def plot_singular_values(self, normalize=True):
        """
        Plot singular values for each layer across different epochs.
        Each curve represents singular values at a particular epoch.
        
        Args:
            normalize: If True, normalize singular values by dividing by the largest singular value,
                      so that each curve starts at 100%.
        """
        if not os.path.exists(self.svd_dir):
            print("No SVD data found.")
            return
        
        # Get all layer directories
        layer_dirs = [d for d in os.listdir(self.svd_dir) if os.path.isdir(os.path.join(self.svd_dir, d))]
        
        for layer_name in layer_dirs:
            layer_dir = os.path.join(self.svd_dir, layer_name)
            
            # Get all CSV files with singular values
            sv_files = [f for f in os.listdir(layer_dir) if f.endswith("_singular_values.csv")]
            
            if not sv_files:
                continue
                
            plt.figure(figsize=(12, 6))
            
            # Sort files by epoch for consistent coloring
            sv_files.sort(key=lambda x: int(x.split('_')[1]))
            
            # Limit to plotting at most 10 epochs to avoid clutter
            max_files_to_plot = 10
            if len(sv_files) > max_files_to_plot:
                step = len(sv_files) // max_files_to_plot
                sv_files = sv_files[::step]
                if sv_files[-1] != sv_files[-1]:  # Ensure the last epoch is included
                    sv_files[-1] = sv_files[-1]
            
            # Generate colormap
            colors = plt.cm.viridis(np.linspace(0, 1, len(sv_files)))
            
            for i, sv_file in enumerate(sv_files):
                # Extract epoch from filename
                epoch = int(sv_file.split('_')[1])
                
                # Load singular values from CSV
                sv_path = os.path.join(layer_dir, sv_file)
                try:
                    # Skip header and read values
                    with open(sv_path, 'r') as f:
                        lines = f.readlines()[1:]  # Skip header
                        singular_values = [float(line.strip().split(',')[1]) for line in lines]
                    
                    # Normalize if requested
                    if normalize and len(singular_values) > 0:
                        max_sv = singular_values[0]  # First value should be the largest
                        if max_sv > 0:
                            singular_values = [sv / max_sv * 100 for sv in singular_values]
                            y_label = 'Relative Singular Value (% of max)'
                    else:
                        y_label = 'Singular Value (log scale)'
                        
                    # Plot singular values in descending order
                    plt.semilogy(range(1, len(singular_values) + 1), singular_values, 
                                color=colors[i], label=f'Epoch {epoch}')
                    
                except Exception as e:
                    print(f"Error reading singular values from {sv_path}: {e}")
            
            plt.xlabel('Index')
            plt.ylabel(y_label)
            title_suffix = ' (Normalized)' if normalize else ''
            plt.title(f'Singular Value Spectrum - Layer: {layer_name}{title_suffix}')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            # Add legend with reasonable size
            if len(sv_files) <= 10:
                plt.legend(loc='upper right')
            else:
                plt.legend(loc='upper right', ncol=2, fontsize='small')
            
            plt.tight_layout()
            plot_type = "normalized" if normalize else "absolute"
            plt.savefig(os.path.join(layer_dir, f"{layer_name}_singular_values_{plot_type}_plot.png"))
            plt.close()
            
            # Always create a version with linear scale to better see the decay pattern
            plt.figure(figsize=(12, 6))
            
            for i, sv_file in enumerate(sv_files):
                epoch = int(sv_file.split('_')[1])
                sv_path = os.path.join(layer_dir, sv_file)
                
                try:
                    with open(sv_path, 'r') as f:
                        lines = f.readlines()[1:]
                        singular_values = [float(line.strip().split(',')[1]) for line in lines]
                    
                    # Apply normalization if requested
                    if normalize and len(singular_values) > 0:
                        max_sv = singular_values[0]
                        if max_sv > 0:
                            singular_values = [sv / max_sv * 100 for sv in singular_values]
                        
                    plt.plot(range(1, len(singular_values) + 1), singular_values, 
                            color=colors[i], label=f'Epoch {epoch}')
                    
                except Exception as e:
                    print(f"Error reading singular values from {sv_path}: {e}")
            
            plt.xlabel('Index')
            
            if normalize:
                plt.ylabel('Relative Singular Value (% of max)')
                plt.title(f'Singular Value Spectrum - Layer: {layer_name} (Normalized, Linear Scale)')
            else:
                plt.ylabel('Singular Value')
                plt.title(f'Singular Value Spectrum - Layer: {layer_name} (Linear Scale)')
                
            plt.grid(True, which="both", ls="-", alpha=0.2)
            
            if len(sv_files) <= 10:
                plt.legend(loc='upper right')
            else:
                plt.legend(loc='upper right', ncol=2, fontsize='small')
            
            plt.tight_layout()
            plot_type = "normalized_linear" if normalize else "absolute_linear"
            plt.savefig(os.path.join(layer_dir, f"{layer_name}_singular_values_{plot_type}_plot.png"))
            plt.close()
    
    def save_checkpoint(self, epoch: int, iteration: int = None):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            iteration: Current iteration number (if within epoch)
        """
        # Create checkpoint identifier
        if iteration is not None:
            identifier = f"epoch_{epoch}_iter_{iteration}"
        else:
            identifier = f"epoch_{epoch}"
        
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else None,
            'val_loss': self.val_losses[-1] if self.val_losses else None,
            'val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None
        }
        
        # Save checkpoint
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_{identifier}.pt"))
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move optimizer state to correct device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        
        if 'train_loss' in checkpoint and checkpoint['train_loss'] is not None:
            self.train_losses.append(checkpoint['train_loss'])
        if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
            self.val_losses.append(checkpoint['val_loss'])
        if 'val_accuracy' in checkpoint and checkpoint['val_accuracy'] is not None:
            self.val_accuracies.append(checkpoint['val_accuracy'])
    
    def train(
        self,
        num_epochs: int,
        eval_interval: int = 1,
        save_activations_interval: int = 1,
        compute_svd_interval: int = 1,
        checkpoint_interval: int = None,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            eval_interval: Interval (in epochs) to evaluate model
            save_activations_interval: Interval (in epochs) to save activations (None = don't save)
            compute_svd_interval: Interval (in epochs) to compute SVD (None = don't compute)
            checkpoint_interval: Interval (in epochs) to save checkpoints (None = only save at end of epoch)
            verbose: Whether to print progress
        """
        start_time = time.time()
        
        # Plot initial singular values
        self.compute_svd(0)
        self.plot_singular_values(normalize=True)
        self.plot_singular_values(normalize=False)

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Save activations at specified intervals
            if save_activations_interval and epoch % save_activations_interval == 0:
                self.save_activations(epoch)
            
            # Compute SVD at specified intervals
            if compute_svd_interval and epoch % compute_svd_interval == 0:
                self.compute_svd(epoch)
                self.plot_singular_values(normalize=True)
                self.plot_singular_values(normalize=False)
            
            # Save checkpoints at specified intervals
            if checkpoint_interval and epoch % checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            
            epoch_time = time.time() - epoch_start
            if verbose:
                if eval_interval and epoch % eval_interval == 0:
                    val_loss, val_acc = self.validate()
                    print(f"Epoch {epoch}/{num_epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_loss:.4f} | "
                          f"Val Acc: {val_acc:.4f} | "
                          f"Time: {epoch_time:.2f}s")
                else:
                    print(f"Epoch {epoch}/{num_epochs} | "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Time: {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Plot training curves
        self.plot_training_curves()
        
        return self.train_losses, self.val_losses, self.val_accuracies
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracy
        if self.val_accuracies:
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, label='Val Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Validation Accuracy')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, 'training_curves.png'))
        plt.close()


class MetaTrainer:
    """
    Meta-trainer for running multiple experiments with different dropout settings.
    """
    def __init__(
        self,
        data_handler_class,
        data_handler_params: Dict,
        model_name: str,
        model_params_base: Dict,
        trainer_params: Dict,
        save_dir: str = './results',
        meta_experiment_name: str = None,
        load: bool = False
    ):
        """
        Initialize the meta trainer.
        
        Args:
            data_handler_class: Class for handling data
            data_handler_params: Parameters for data handler
            model_name: Name of the model to use
            model_params_base: Base parameters for the model (excluding dropout)
            trainer_params: Parameters for the trainer
            save_dir: Directory to save results
            meta_experiment_name: Name for the meta experiment
            load: If True, load results from the meta_experiment_name directory
                  instead of creating a new experiment
        """
        self.data_handler_class = data_handler_class
        self.data_handler_params = data_handler_params
        self.model_name = model_name
        self.model_params_base = model_params_base
        self.trainer_params = trainer_params
        self.save_dir = save_dir
        
        # Generate meta experiment name if not provided
        if meta_experiment_name is None:
            self.meta_experiment_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.meta_experiment_name = meta_experiment_name
            
        # Create meta experiment directory
        self.meta_dir = os.path.join(save_dir, self.meta_experiment_name)
        
        # Store results
        self.results = {}
        
        print("meta exp name: ", self.meta_experiment_name, "meta dir: ", self.meta_dir)

        # If load is True, load existing results
        if load:
            if not os.path.exists(self.meta_dir):
                raise FileNotFoundError(f"Results directory {self.meta_dir} not found.")
            self.load_results(self.meta_dir)
        else:
            os.makedirs(self.meta_dir, exist_ok=True)
            print("ln 699: making dir ", self.meta_dir)
        
    def run_experiment(
        self,
        dropout_ratios: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        num_seeds: int = 3,
        train_params: Dict = None
    ):
        """
        Run multiple experiments with different dropout ratios and seeds.
        
        Args:
            dropout_ratios: List of dropout probabilities to try
            num_seeds: Number of seeds to use for each dropout ratio
            train_params: Parameters for training
        """
        if train_params is None:
            train_params = {
                'num_epochs': 10,
                'eval_interval': 1,
                'compute_svd_interval': 500,
                'checkpoint_interval': 2000,
                'verbose': True
            }
        
        # Generate different seeds
        seeds = list(range(42, 42 + num_seeds))
        
        from models.architectures import get_model
        
        # Run experiments for each dropout ratio and seed
        for dropout_ratio in dropout_ratios:
            dropout_results = []
            
            # Create data handler for first seed to get shapes
            data_params = self.data_handler_params.copy()
            data_params['seed'] = seeds[0]
            data_handler = self.data_handler_class(**data_params)
            
            # Get loaders 
            train_loader, val_loader, test_loader = data_handler.load_data()
            
            # Get input and output shapes
            input_shape = data_handler.get_input_shape()
            output_shape = data_handler.get_output_shape()
            
            # Create model with specific dropout ratio
            model_params = self.model_params_base.copy()
            model_params['dropout_prob'] = dropout_ratio
            
            # Create model for architecture printing (first seed)
            model = get_model(
                model_name=self.model_name,
                input_shape=input_shape,
                output_dim=output_shape,
                model_params=model_params
            )
            
            # Print network architecture, input shape, and output shape once per dropout setting
            print(f"\n{'='*80}")
            print(f"NETWORK CONFIGURATION FOR DROPOUT = {dropout_ratio}")
            print(f"{'='*80}")
            print(f"Model: {self.model_name}")
            print(f"Input shape: {input_shape}")
            print(f"Output shape: {output_shape}")
            print(f"Model parameters: {model_params}")
            print(f"Network Architecture:")
            print(f"{model}")
            print(f"{'='*80}\n")
            
            for seed in seeds:
                print(f"\n{'-'*80}")
                print(f"Running experiment with dropout={dropout_ratio}, seed={seed}")
                print(f"{'-'*80}\n")
                
                # Create data handler with specific seed
                if seed != seeds[0]:  # Reuse the data handler for first seed
                    data_params = self.data_handler_params.copy()
                    data_params['seed'] = seed
                    data_handler = self.data_handler_class(**data_params)
                    train_loader, val_loader, test_loader = data_handler.load_data()
                
                # Create model for this specific seed if not the first seed
                if seed != seeds[0]:
                    model = get_model(
                        model_name=self.model_name,
                        input_shape=input_shape,
                        output_dim=output_shape,
                        model_params=model_params
                    )
                
                # Create experiment name
                experiment_name = f"dropout_{dropout_ratio}_seed_{seed}"
                
                # Setup trainer
                trainer_params = self.trainer_params.copy()
                trainer_params.update({
                    'model': model,
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'test_loader': test_loader,
                    'save_dir': self.meta_dir,
                    'experiment_name': experiment_name,
                    'seed': seed
                })
                
                trainer = Trainer(**trainer_params)
                
                # Train model
                train_losses, val_losses, val_accuracies = trainer.train(**train_params)
                
                # Evaluate on test set
                test_loss, test_acc = trainer.validate(test_loader)
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
                
                # Store results
                exp_result = {
                    'dropout_ratio': dropout_ratio,
                    'seed': seed,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_accuracies': val_accuracies,
                    'test_loss': test_loss,
                    'test_accuracy': test_acc,
                    'experiment_name': experiment_name
                }
                
                dropout_results.append(exp_result)
            
            # Store results for this dropout ratio
            self.results[dropout_ratio] = dropout_results
        
        # Save and plot aggregate results
        self.save_results()
        self.plot_results()
        
        return self.results
    
    def load_results(self, results_dir: str):
        """
        Load results from an existing experiment directory.
        
        Args:
            results_dir: Path to the experiment directory containing meta_results.json
        """
        results_file = os.path.join(results_dir, 'meta_results.json')
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Results file {results_file} not found.")
        
        # Load results from file
        with open(results_file, 'r') as f:
            serialized_results = json.load(f)
        
        # Convert string dropout ratios back to float
        self.results = {}
        for dropout_ratio_str, exp_results in serialized_results.items():
            dropout_ratio = float(dropout_ratio_str)
            self.results[dropout_ratio] = exp_results
        
        # Update meta_dir to the loaded results directory
        self.meta_dir = results_dir
        
        return self.results
    
    def save_results(self):
        """Save experiment results to file"""
        results_file = os.path.join(self.meta_dir, 'meta_results.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for dropout_ratio, exp_results in self.results.items():
            serializable_exp_results = []
            
            for exp_result in exp_results:
                serializable_exp_result = {}
                for k, v in exp_result.items():
                    if isinstance(v, np.ndarray):
                        serializable_exp_result[k] = v.tolist()
                    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                        serializable_exp_result[k] = [arr.tolist() for arr in v]
                    else:
                        serializable_exp_result[k] = v
                
                serializable_exp_results.append(serializable_exp_result)
            
            serializable_results[str(dropout_ratio)] = serializable_exp_results
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=4)
    
    def plot_results(self, results_dir: str = None):
        """
        Plot aggregate results across dropout ratios.
        
        Args:
            results_dir: Optional directory containing results to plot.
                         If provided, results will be loaded from this directory.
                         If None, the current results in self.results will be used.
        """
        # If results_dir is provided, load results from that directory
        if results_dir is not None:
            self.load_results(results_dir)
            
        if not self.results:
            print("No results to plot.")
            return
        
        # Create figure for test accuracy vs dropout ratio
        plt.figure(figsize=(10, 6))
        
        dropout_ratios = sorted(self.results.keys())
        test_accuracies = []
        test_accuracies_std = []
        
        for dropout_ratio in dropout_ratios:
            accuracies = [exp['test_accuracy'] for exp in self.results[dropout_ratio]]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            
            test_accuracies.append(mean_acc)
            test_accuracies_std.append(std_acc)
        
        plt.errorbar(
            dropout_ratios, 
            test_accuracies, 
            yerr=test_accuracies_std, 
            marker='o', 
            linestyle='-',
            capsize=5
        )
        
        plt.xlabel('Dropout Ratio')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs. Dropout Ratio')
        plt.grid(True)
        
        plt.savefig(os.path.join(self.meta_dir, 'test_accuracy_vs_dropout.png'))
        plt.close()
        
        # Create figure for validation accuracy curves
        plt.figure(figsize=(15, 10))
        
        for i, dropout_ratio in enumerate(dropout_ratios):
            plt.subplot(2, 3, i+1)
            
            for exp_result in self.results[dropout_ratio]:
                plt.plot(
                    exp_result['val_accuracies'],
                    label=f"Seed {exp_result['seed']}"
                )
            
            plt.title(f"Dropout = {dropout_ratio}")
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy')
            plt.grid(True)
            
            if i == 0:
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.meta_dir, 'val_accuracy_curves.png'))
        plt.close()
        
        # Plot singular values across dropout rates
        self.plot_singular_values_across_dropout()
    
    def plot_singular_values_across_dropout(self):
        """
        Plot singular values of the final epoch for different dropout rates.
        Each curve represents singular values for a specific dropout rate.
        """
        if not self.results:
            return
        
        dropout_ratios = sorted(self.results.keys())
        
        # Get all experiment directories
        all_experiment_dirs = []
        for dropout_ratio in dropout_ratios:
            for exp_result in self.results[dropout_ratio]:
                exp_dir = os.path.join(self.meta_dir, exp_result['experiment_name'])
                if os.path.exists(exp_dir):
                    all_experiment_dirs.append((dropout_ratio, exp_dir))
        
        # First, find all layers that have SVD data
        all_layers = set()
        for _, exp_dir in all_experiment_dirs:
            svd_dir = os.path.join(exp_dir, 'svd')
            if os.path.exists(svd_dir):
                layers = [d for d in os.listdir(svd_dir) if os.path.isdir(os.path.join(svd_dir, d))]
                all_layers.update(layers)
        
        if not all_layers:
            print("No SVD data found across experiments.")
            return
        
        # For each layer, create a plot with singular values across dropout rates
        for layer_name in all_layers:
            plt.figure(figsize=(4, 4))
            
            # Generate colormap for dropout rates
            colors = plt.cm.viridis(np.linspace(0, 1, len(dropout_ratios)))
            
            # Collect data for this layer across dropout rates
            for i, dropout_ratio in enumerate(dropout_ratios):
                # Average across seeds for this dropout ratio
                all_singular_values = []
                
                for dropout_ratio_actual, exp_dir in all_experiment_dirs:
                    if dropout_ratio_actual != dropout_ratio:
                        continue
                        
                    svd_layer_dir = os.path.join(exp_dir, 'svd', layer_name)
                    if not os.path.exists(svd_layer_dir):
                        continue
                    
                    # Find the last epoch's singular values
                    sv_files = [f for f in os.listdir(svd_layer_dir) if f.endswith("_singular_values.csv")]
                    if not sv_files:
                        continue
                        
                    # Extract epoch numbers and find the highest
                    epochs = [int(f.split('_')[1]) for f in sv_files if f.split('_')[0] == 'epoch']
                    if not epochs:
                        continue
                        
                    max_epoch = max(epochs)
                    last_epoch_file = f"epoch_{max_epoch}_singular_values.csv"
                    
                    # Try alternative format if file not found
                    if not os.path.exists(os.path.join(svd_layer_dir, last_epoch_file)):
                        matching_files = [f for f in sv_files if f.startswith(f"epoch_{max_epoch}_")]
                        if matching_files:
                            last_epoch_file = matching_files[0]
                        else:
                            continue
                    
                    # Read singular values
                    try:
                        sv_path = os.path.join(svd_layer_dir, last_epoch_file)
                        with open(sv_path, 'r') as f:
                            lines = f.readlines()[1:]  # Skip header
                            singular_values = [float(line.strip().split(',')[1]) for line in lines]
                        all_singular_values.append(singular_values)
                    except Exception as e:
                        print(f"Error reading singular values from {sv_path}: {e}")
                
                # If we have data for this dropout ratio, average and plot it
                if all_singular_values:
                    # Find the minimum length to align all arrays
                    min_length = min(len(sv) for sv in all_singular_values)
                    
                    # Truncate all arrays to the same length and compute average
                    aligned_svs = [sv[:min_length] for sv in all_singular_values]
                    avg_singular_values = np.mean(aligned_svs, axis=0)
                    
                    # Plot on both log and linear scales
                    plt.semilogy(
                        range(1, len(avg_singular_values) + 1), 
                        avg_singular_values, 
                        color=colors[i], 
                        label=f'Dropout = {dropout_ratio}'
                    )
            
            # Clean up layer name for plot title
            layer_name_plt = layer_name.replace("layer", "").strip()
            layer_name_plt = layer_name_plt.replace("network", "").strip()

            plt.xlabel('Index')
            plt.ylabel('Singular Value')
            # plt.title(f'Singular Value of Activations at Layer {layer_name.split(".")[-1]}')
            plt.title(f'Singular Value of Activations at Layer {layer_name_plt}')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(self.meta_dir, f"{layer_name}_singular_values_by_dropout.pdf"))
            plt.close()
            
            # Also create a linear scale plot
            plt.figure(figsize=(4, 4))
            
            for i, dropout_ratio in enumerate(dropout_ratios):
                all_singular_values = []
                
                for dropout_ratio_actual, exp_dir in all_experiment_dirs:
                    if dropout_ratio_actual != dropout_ratio:
                        continue
                        
                    svd_layer_dir = os.path.join(exp_dir, 'svd', layer_name)
                    if not os.path.exists(svd_layer_dir):
                        continue
                    
                    # Find the last epoch's singular values
                    sv_files = [f for f in os.listdir(svd_layer_dir) if f.endswith("_singular_values.csv")]
                    if not sv_files:
                        continue
                        
                    # Extract epoch numbers and find the highest
                    epochs = [int(f.split('_')[1]) for f in sv_files if f.split('_')[0] == 'epoch']
                    if not epochs:
                        continue
                        
                    max_epoch = max(epochs)
                    last_epoch_file = f"epoch_{max_epoch}_singular_values.csv"
                    
                    # Try alternative format if file not found
                    if not os.path.exists(os.path.join(svd_layer_dir, last_epoch_file)):
                        matching_files = [f for f in sv_files if f.startswith(f"epoch_{max_epoch}_")]
                        if matching_files:
                            last_epoch_file = matching_files[0]
                        else:
                            continue
                    
                    # Read singular values
                    try:
                        sv_path = os.path.join(svd_layer_dir, last_epoch_file)
                        with open(sv_path, 'r') as f:
                            lines = f.readlines()[1:]  # Skip header
                            singular_values = [float(line.strip().split(',')[1]) for line in lines]
                        all_singular_values.append(singular_values)
                    except Exception as e:
                        print(f"Error reading singular values from {sv_path}: {e}")
                
                # If we have data for this dropout ratio, average and plot it
                if all_singular_values:
                    # Find the minimum length to align all arrays
                    min_length = min(len(sv) for sv in all_singular_values)
                    
                    # Truncate all arrays to the same length and compute average
                    aligned_svs = [sv[:min_length] for sv in all_singular_values]
                    avg_singular_values = np.mean(aligned_svs, axis=0)
                    
                    # Plot with linear scale
                    plt.plot(
                        range(1, len(avg_singular_values) + 1), 
                        avg_singular_values, 
                        color=colors[i], 
                        label=f'Dropout = {dropout_ratio}'
                    )

            plt.xlabel('Index')
            plt.ylabel('Singular Value')
            plt.title(f'Singular Value of Activations at Layer {layer_name_plt}')
            plt.grid(True)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(self.meta_dir, f"{layer_name}_singular_values_by_dropout_linear.pdf"))
            plt.close()