import torch
import numpy as np
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, Dict, Any


class DataHandler:
    """
    Class for loading and handling various datasets (MNIST, CIFAR10, synthetic)
    """
    def __init__(self, 
                 dataset_name: str = "mnist", 
                 batch_size: int = 128, 
                 num_workers: int = 2, 
                 data_dir: str = './data',
                 val_split: float = 0.1,
                 seed: int = 42):
        """
        Initialize data handler
        
        Args:
            dataset_name: Name of dataset ('mnist', 'cifar10', 'synthetic')
            batch_size: Batch size for data loader
            num_workers: Number of workers for data loading
            data_dir: Directory to store/load data
            val_split: Validation split ratio (0-1)
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.val_split = val_split
        self.seed = seed
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        torch.manual_seed(seed)
        np.random.seed(seed)

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load the specified dataset
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.dataset_name == "mnist":
            return self._load_mnist()
        elif self.dataset_name == "cifar10":
            return self._load_cifar10()
        elif self.dataset_name == "synthetic":
            return self._load_synthetic()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_mnist(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load MNIST dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Load training data
        train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=transform)
        
        # Split into train and validation
        train_size = int((1 - self.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = data.random_split(
            train_dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Load test data
        test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def _load_cifar10(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load CIFAR10 dataset"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        # Load training data
        train_dataset = datasets.CIFAR10(self.data_dir, train=True, download=True, transform=transform_train)
        
        # Split into train and validation
        train_size = int((1 - self.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = data.random_split(
            train_dataset, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Set transform for validation dataset
        val_dataset.dataset.transform = transform_test
        
        # Load test data
        test_dataset = datasets.CIFAR10(self.data_dir, train=False, download=True, transform=transform_test)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        return self.train_loader, self.val_loader, self.test_loader

    def create_synthetic_data(
        self, 
        input_dim: int = 20, 
        output_dim: int = 20, 
        n_samples: int = 10000,
        noise_std: float = 0.1,
        data_type: str = "linear",
        rank: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create synthetic data
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_samples: Number of samples
            noise_std: Standard deviation of Gaussian noise
            data_type: Type of synthetic data ('linear', 'nonlinear')
            rank: Rank of the weight matrix W (default: 15)
            
        Returns:
            Tuple of (inputs, targets)
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Generate input data
        X = torch.randn(n_samples, input_dim)
        
        # Generate true weights with specified rank
        if rank < min(input_dim, output_dim):
            # Create low-rank weight matrix
            U = torch.randn(input_dim, rank)
            V = torch.randn(rank, output_dim)
            W = U @ V
        else:
            # Use full-rank weight matrix
            W = torch.randn(input_dim, output_dim)
        
        # Generate output data based on data type
        if data_type == "linear":
            # Y = XW + noise
            Y = X @ W + noise_std * torch.randn(n_samples, output_dim)
        elif data_type == "nonlinear":
            # Y = sin(XW) + noise
            Y = torch.sin(X @ W) + noise_std * torch.randn(n_samples, output_dim)
        else:
            raise ValueError(f"Unknown synthetic data type: {data_type}")
            
        return X, Y

    def _load_synthetic(
        self, 
        input_dim: int = 20, 
        output_dim: int = 20, 
        n_samples: int = 10000,
        noise_std: float = 0.0,
        data_type: str = "linear",
        rank: int = 15
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load synthetic dataset by generating a single dataset and splitting it
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_samples: Number of samples
            noise_std: Standard deviation of Gaussian noise
            data_type: Type of synthetic data ('linear', 'nonlinear')
            rank: Rank of the weight matrix W (default: 15)
        """
        # Create single synthetic dataset then split
        X, y = self.create_synthetic_data(
            input_dim, output_dim, n_samples, noise_std, data_type, rank
        )
        
        # Split into train, validation, and test
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)
        n_test = n_samples - n_train - n_val
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers
        )
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_input_shape(self) -> Tuple:
        """Get input shape of the dataset"""
        if self.train_loader is None:
            self.load_data()
            
        # Get the first batch
        for x, _ in self.train_loader:
            return x.shape[1:]  # Return shape without batch dimension
            
    def get_output_shape(self) -> int:
        """Get output shape/classes of the dataset"""
        if self.dataset_name == "mnist":
            return 10
        elif self.dataset_name == "cifar10":
            return 10
        elif self.dataset_name == "synthetic":
            if self.train_loader is None:
                self.load_data()
            
            # Get the first batch
            for _, y in self.train_loader:
                return y.shape[1]  # Return output dimension
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")