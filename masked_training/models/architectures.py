import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class ActivationExtractor(nn.Module):
    """
    Wrapper module to extract activations from a network.
    This wrapper enables saving activations from specified layers during forward pass.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.activations = {}
        self.hooks = []
        self.layer_names = []
        
    def register_hooks(self, layer_names: Union[List[str], str] = None):
        """
        Register hooks for the specified layers
        
        Args:
            layer_names: List of layer names to extract activations from,
                         or 'all' to extract from all layers
        """
        # Clear existing hooks
        self.remove_hooks()
        self.activations = {}
        self.layer_names = []
        
        # If layer_names is None or empty, don't register any hooks
        if not layer_names:
            return
        
        # If 'all', extract from all layers with parameters
        if layer_names == 'all':
            for name, module in self.model.named_modules():
                if list(module.parameters()):  # Only register hook if module has parameters
                    self._register_hook(name, module)
        else:
            # Register hooks for specified layers
            for name, module in self.model.named_modules():
                if name in layer_names:
                    self._register_hook(name, module)
    
    def _register_hook(self, name: str, module: nn.Module):
        """Register a hook for the specified module"""
        def hook_fn(module, input, output):
            self.activations[name] = output.detach()
            
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        self.layer_names.append(name)
        
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation extraction"""
        return self.model(x)
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get the extracted activations"""
        return self.activations


class DeepLinearNetwork(nn.Module):
    """
    Deep Linear Network with configurable depth and width
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [128, 128],
        dropout_prob: float = 0.0,
        dropout_locations: List[int] = None  # Layer indices where dropout should be applied
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        
        # Default: apply dropout after each hidden layer
        if dropout_locations is None:
            self.dropout_locations = list(range(len(hidden_dims)))
        else:
            self.dropout_locations = dropout_locations
            
        # Build layers
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f'linear{i}', nn.Linear(prev_dim, hidden_dim, bias=False)))
            
            # Initialize weights using uniform distribution
            stdv = 1. / (prev_dim * hidden_dim)**0.5 
            # layers[-1][1].weight.data.uniform_(-stdv, stdv)
            # print('stdv:', stdv)
            
            
            if i in self.dropout_locations:
                layers.append((f'dropout{i}', nn.Dropout(p=dropout_prob)))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append((f'linear{len(hidden_dims)}', nn.Linear(prev_dim, output_dim, bias=False)))
        
        # Initialize output layer weights
        stdv = 1. / (prev_dim * output_dim)**0.5 
        # layers[-1][1].weight.data.uniform_(-stdv, stdv)
        # print('stdv:', stdv)
        
        # Create the network
        self.network = nn.Sequential(
            *[module for _, module in layers]
        )
        
        # Store named modules for hook registration
        self.named_layers = dict(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        return self.network(x)


class MLP(nn.Module):
    """
    Multi-layer Perceptron with configurable depth, width and activation function
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_dims: List[int] = [128, 128],
        activation: str = 'relu',
        dropout_prob: float = 0.0,
        dropout_locations: List[int] = None  # Layer indices where dropout should be applied
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout_prob = dropout_prob
        
        # Default: apply dropout after each hidden layer
        if dropout_locations is None:
            self.dropout_locations = list(range(len(hidden_dims)))
        else:
            self.dropout_locations = dropout_locations
        
        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        layers = []
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append((f'linear{i}', nn.Linear(prev_dim, hidden_dim)))
            layers.append((f'activation{i}', self.activation))
            
            if i in self.dropout_locations:
                layers.append((f'dropout{i}', nn.Dropout(p=dropout_prob)))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append((f'linear{len(hidden_dims)}', nn.Linear(prev_dim, output_dim)))
        
        # Create the network
        self.network = nn.Sequential(
            *[module for _, module in layers]
        )
        
        # Store named modules for hook registration
        self.named_layers = dict(layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        return self.network(x)


class ResidualBlock(nn.Module):
    """
    Residual block for ResNet
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_prob: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_prob)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        
        if self.dropout1:
            out = self.dropout1(out)
            
        out = self.bn2(self.conv2(out))
        
        if self.dropout2:
            out = self.dropout2(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture with configurable depth and dropout
    """
    def __init__(
        self,
        num_blocks: List[int] = [2, 2, 2, 2],  # ResNet18 by default
        in_channels: int = 3,
        num_classes: int = 10,
        dropout_prob: float = 0.0,
        include_top: bool = True
    ):
        super().__init__()
        
        self.in_planes = 64
        self.dropout_prob = dropout_prob
        self.include_top = include_top
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        
        # Classification head
        if include_top:
            self.linear = nn.Linear(512, num_classes)
            
    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(
                ResidualBlock(
                    self.in_planes, planes, stride, 
                    dropout_prob=self.dropout_prob
                )
            )
            self.in_planes = planes
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        
        if self.include_top:
            out = self.linear(out)
            
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feed-forward network
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        
        # Feed-forward network
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class SimpleTransformer(nn.Module):
    """
    Simple Transformer model with configurable parameters
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 28,  # For flattened images like MNIST (28x28)
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Position encoding - learnable
        self.position_embedding = nn.Parameter(torch.zeros(seq_len, 1, d_model))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.out = nn.Linear(d_model * seq_len, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input: [batch_size, channels, height, width] -> [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        
        if x.dim() > 2:
            if x.dim() == 4:  # Image input [B, C, H, W]
                # For images, we reshape to [B, H*W, C]
                x = x.view(batch_size, x.size(1), -1)
                x = x.permute(0, 2, 1)
            else:  # Already 3D tensor
                pass
        else:
            # For 2D input [B, features], reshape to [B, seq_len, features/seq_len]
            assert x.size(1) % self.seq_len == 0, "Input dimension must be divisible by seq_len"
            x = x.view(batch_size, self.seq_len, -1)
        
        # Apply embedding
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        
        # Add position embedding
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, d_model]
        x = x + self.position_embedding
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Reshape and apply output layer
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, d_model]
        x = x.reshape(batch_size, -1)  # [batch_size, seq_len * d_model]
        x = self.dropout(x)
        x = self.out(x)
        
        return x


def get_model(
    model_name: str,
    input_shape: Tuple,
    output_dim: int,
    model_params: Dict = None
) -> nn.Module:
    """
    Factory function to create a model based on name and parameters
    
    Args:
        model_name: Name of the model ('deeplinear', 'mlp', 'resnet', 'transformer')
        input_shape: Shape of input tensor (excluding batch dimension)
        output_dim: Dimension of output (number of classes)
        model_params: Additional model parameters
        
    Returns:
        Instantiated model
    """
    # Default parameters
    default_params = {
        'deeplinear': {
            'hidden_dims': [128, 128],
            'dropout_prob': 0.0,
            'dropout_locations': None
        },
        'mlp': {
            'hidden_dims': [128, 128],
            'activation': 'relu',
            'dropout_prob': 0.0,
            'dropout_locations': None
        },
        'resnet': {
            'num_blocks': [2, 2, 2, 2],  # ResNet18
            'dropout_prob': 0.0,
            'include_top': True
        },
        'transformer': {
            'seq_len': 28,  # Default for MNIST
            'd_model': 128,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 512,
            'dropout': 0.0
        }
    }
    
    # Update default parameters with provided ones
    params = default_params.get(model_name, {}).copy()
    if model_params:
        # Filter model_params to only include keys that are valid for this model type
        filtered_params = {k: v for k, v in model_params.items() if k in params}
        params.update(filtered_params)
    
    # Calculate input dimension
    if len(input_shape) > 1:
        input_dim = input_shape[0] if len(input_shape) == 1 else input_shape[0] * input_shape[1]
        if len(input_shape) > 2:
            input_dim *= input_shape[2]
    else:
        input_dim = input_shape[0]
    
    # Create model
    if model_name == 'deeplinear':
        model = DeepLinearNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            **params
        )
    elif model_name == 'mlp':
        model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            **params
        )
    elif model_name == 'resnet':
        in_channels = input_shape[0] if len(input_shape) > 2 else 1
        model = ResNet(
            in_channels=in_channels,
            num_classes=output_dim,
            **params
        )
    elif model_name == 'transformer':
        if len(input_shape) > 1:
            # For images or 3D inputs
            params['seq_len'] = input_shape[1] * (input_shape[2] if len(input_shape) > 2 else 1)
            input_dim = input_shape[0]  # Number of channels
        else:
            input_dim = input_shape[0] // params['seq_len']
            
        model = SimpleTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            **params
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Wrap with activation extractor
    model = ActivationExtractor(model)
    
    return model