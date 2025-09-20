import argparse
import os
import torch
import torch.nn as nn
import numpy as np

from data.data_loader import DataHandler
from models.architectures import get_model
from utils.training import Trainer, MetaTrainer, get_loss_function


def parse_args():
    parser = argparse.ArgumentParser(description='Dropout Effect on Singular Values')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['mnist', 'cifar10', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to store/load data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='deeplinear', choices=['deeplinear', 'mlp', 'resnet', 'transformer'],
                        help='Model architecture')
    # parser.add_argument('--hidden_dims', type=str, default='256,256',
    parser.add_argument('--hidden_dims', type=str, default='20,20',
                        help='Hidden dimensions, comma-separated')
    parser.add_argument('--activation', type=str, default='relu', 
                        choices=['relu', 'tanh', 'sigmoid', 'leaky_relu'],
                        help='Activation function (for MLP)')
    
    # Loss function parameter
    parser.add_argument('--loss', type=str, default='mse', 
                        choices=['mse', 'crossentropy', 'bce', 'l1', 'smoothl1'],
                        help='Loss function to use for training')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Interval (in epochs) to evaluate model')
    parser.add_argument('--compute_svd_interval', type=int, default=2,
                        help='Interval (in iterations) to compute SVD')
    parser.add_argument('--checkpoint_interval', type=int, default=None,
                        help='Interval (in iterations) to save checkpoints')
    parser.add_argument('--save_activations_interval', type=int, default=None, 
                        help='Interval (in iterations) to save activations')
    
    # Dropout parameters
    parser.add_argument('--dropout_prob', type=float, default=0.0,
                        help='Dropout probability (for single run)')
    parser.add_argument('--track_layers', type=str, default='all',
                        help='Layers to track activations from (comma-separated or "all")')
    
    # Meta-experiment parameters
    parser.add_argument('--meta_experiment', action='store_true',
                        help='Run meta experiment with multiple dropout ratios and seeds')
    parser.add_argument('--dropout_ratios', type=str, default='0.0,0.3,0.6,0.9',
                        help='Dropout ratios for meta experiment (comma-separated)')
    parser.add_argument('--num_seeds', type=int, default=3,
                        help='Number of seeds per dropout ratio')
    
    # Output parameters
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name for the experiment')
    
    # Evaluation-only parameters
    parser.add_argument('--eval', type=str, default=None,
                        help='Path to results directory for evaluation-only mode (no training)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set up directory structure
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    np.random.seed(args.seed)
    
    # If in evaluation-only mode
    if args.eval is not None:
        print(f"Evaluation-only mode: Loading results from {args.eval}")
        if not os.path.exists(args.eval):
            raise FileNotFoundError(f"Results directory {args.eval} not found.")
        
        # Create a MetaTrainer instance with load=True to load existing results
        meta_trainer = MetaTrainer(
            data_handler_class=None,  # Not needed for evaluation only
            data_handler_params={},
            model_name="",  # Not needed for evaluation only
            model_params_base={},
            trainer_params={},
            save_dir=args.save_dir,
            meta_experiment_name=os.path.basename(args.eval),
            load=True
        )
        
        # Plot results - no need to pass results_dir since it's already loaded
        meta_trainer.plot_results()
        print(f"Results successfully plotted to {args.eval}")
        return
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Parse track layers
    if args.track_layers == 'all' or not args.track_layers:
        track_layers = 'all'
    else:
        track_layers = [layer.strip() for layer in args.track_layers.split(',')]
    
    # Get loss function
    criterion = get_loss_function(args.loss)
    
    # Data handler parameters
    data_handler_params = {
        'dataset_name': args.dataset,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'data_dir': args.data_dir,
        'seed': args.seed
    }
    
    # Model parameters
    model_params = {
        'hidden_dims': hidden_dims,
    }
    
    # Add model-specific parameters
    if args.model == 'mlp':
        model_params['activation'] = args.activation
    
    if not args.meta_experiment:
        # Single experiment
        model_params['dropout_prob'] = args.dropout_prob
        
        # Create data handler
        data_handler = DataHandler(**data_handler_params)
        train_loader, val_loader, test_loader = data_handler.load_data()
        
        # Get input and output shapes
        input_shape = data_handler.get_input_shape()
        output_shape = data_handler.get_output_shape()
        
        # Create model
        model = get_model(
            model_name=args.model,
            input_shape=input_shape,
            output_dim=output_shape,
            model_params=model_params
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,  # Pass the loss function
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir,
            experiment_name=args.experiment_name,
            track_layers=track_layers,
            seed=args.seed
        )
        
        # Train model
        trainer.train(
            num_epochs=args.epochs,
            eval_interval=args.eval_interval,
            compute_svd_interval=args.compute_svd_interval,
            checkpoint_interval=args.checkpoint_interval,
            save_activations_interval=args.save_activations_interval,
            verbose=True
        )
        
        # Evaluate on test set
        test_loss, test_acc = trainer.validate(test_loader)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    else:
        # Meta experiment with multiple dropout ratios and seeds
        dropout_ratios = [float(ratio) for ratio in args.dropout_ratios.split(',')]
        
        # Create meta trainer
        meta_trainer = MetaTrainer(
            data_handler_class=DataHandler,
            data_handler_params=data_handler_params,
            model_name=args.model,
            model_params_base=model_params,
            trainer_params={
                'lr': args.lr,
                'device': args.device,
                'track_layers': track_layers,
                'criterion': criterion  # Pass the loss function
            },
            save_dir=args.save_dir,
            meta_experiment_name=args.experiment_name
        )
        
        # Run meta experiment
        meta_trainer.run_experiment(
            dropout_ratios=dropout_ratios,
            num_seeds=args.num_seeds,
            train_params={
                'num_epochs': args.epochs,
                'eval_interval': args.eval_interval,
                'compute_svd_interval': args.compute_svd_interval,
                'checkpoint_interval': args.checkpoint_interval,
                'save_activations_interval': args.save_activations_interval,
                'verbose': True
            }
        )


if __name__ == '__main__':
    main()