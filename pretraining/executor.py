import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utility import Utils, DataGenerator
from model import MLP
from log import Logger

parser = argparse.ArgumentParser(description='M4 Pretraining Script')

# Data arguments
parser.add_argument('--data_folder', type=str, default='/raid/sepideh/Project_MCL/1D-AEMpy-UW-metabolism-BM',
                    help='Path to data folder')
parser.add_argument('--data_file', type=str, default='all_data_lake_modeling_process_based.csv',
                    help='Name of data file')
parser.add_argument('--config_file', type=str, default='../config.json',
                    help='Path to config JSON file')
parser.add_argument('--module', type=str, default='m4',
                    help='Module number from config file')

# Model arguments
parser.add_argument('--hidden_layers', type=int, nargs='+', default=[32, 32],
                    help='Hidden layer sizes')
parser.add_argument('--activation', type=str, default='gelu',
                    choices=['relu', 'tanh', 'gelu'],
                    help='Activation function')
parser.add_argument('--initialization', type=str, default='xavier',
                    choices=['xavier', 'kaiming'],
                    help='Weight initialization method')

# Training arguments
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of training epochs')
parser.add_argument('--decay_steps', type=int, default=500,
                    help='Learning rate decay steps')
parser.add_argument('--decay_rate', type=float, default=0.1,
                    help='Learning rate decay rate')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Dropout rate for regularization')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay for L2 regularization')


# Device arguments
parser.add_argument('--cuda_device', type=int, default=4, help='CUDA device number')

# Output arguments
parser.add_argument('--model_save_path', type=str, default='../saved_models/pretrain',
                    help='Path to save the trained model')

# Wandb arguments
parser.add_argument('--wandb_project', type=str, default='mcl-pretraining',
                    help='Wandb project name')
parser.add_argument('--wandb_run_name', type=str, default="test",
                    help='Wandb run name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Wandb entity/username')


# Setup environment
args = parser.parse_args()

# Initialize wandb
wandb_config = {
    'data_folder': args.data_folder,
    'data_file': args.data_file,
    'config_file': args.config_file,
    'module': args.module,
    'hidden_layers': args.hidden_layers,
    'activation': args.activation,
    'initialization': args.initialization,
    'batch_size': args.batch_size,
    'learning_rate': args.learning_rate,
    'num_epochs': args.num_epochs,
    'decay_steps': args.decay_steps,
    'decay_rate': args.decay_rate,
    'dropout_rate': args.dropout_rate,
    'weight_decay': args.weight_decay,
    'cuda_device': args.cuda_device,
    'model_save_path': args.model_save_path,
}

wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    entity=args.wandb_entity,
    config=wandb_config
)

def main():
    utils = Utils(args.config_file, args.cuda_device, args.data_folder, args.data_file, args.module)
    
    # Set random seeds for reproducibility
    utils.set_random_seed(2025)
    
    logger = Logger(utils)
    
    # Load data
    print("Loading data...")
    file_path = os.path.join(args.data_folder, args.data_file)
    df = pd.read_csv(file_path)
    
    # Select relevant columns first
    all_columns = utils.model_config['input_columns'] + utils.model_config['output_columns']
    df_m = df[all_columns]
    print(f"Data shape: {df_m.shape}")
    
    
    
    # Log data info to wandb
    logger.log_data_info(df_m)
    
    # Prepare data
    train_df_unnrm, val_df_unnrm, test_df_unnrm = utils.split_data(df_m)
    utils.store_datetime_depth_column(df)

    # Normalize features
    train_df, val_df, test_df, scaler = utils.normalize_features(train_df_unnrm, val_df_unnrm, test_df_unnrm)
    train_loader, val_loader, test_loader = utils.data_loaders(train_df, val_df, test_df, args.batch_size)
    
    
    print("Creating model...")
    input_size = train_loader.dataset.X.shape[-1]
    output_size = train_loader.dataset.Y.shape[-1]
    layers = [input_size] + args.hidden_layers + [output_size]
    model = MLP(layers, activation=args.activation, init=args.initialization, dropout_rate=args.dropout_rate)
    model = model.to(utils.device)
    print(model)
    
    # Training
    print("Starting training...")
    train_loss, val_loss, test_loss = model.run_training(
        train_loader, val_loader, test_loader, args, utils.device, logger
    )
    
    # Plot training curves and log to wandb
    logger.log_training_curves(train_loss, val_loss, test_loss)
    print(f"  Final train loss: {train_loss[-1]:.6f}")
    print(f"  Final validation loss: {val_loss[-1]:.6f}")
    print(f"  Final test loss: {test_loss[-1]:.6f}")


    # Save model
    save_dir = os.path.join(args.model_save_path, f"{args.module}_test.pth")
    utils.save_model(model, save_dir)
    

    # Evaluate
    train_avg_loss, train_all_predictions, train_all_targets = model.evaluate(train_loader, utils.device, return_predictions=True)
    val_avg_loss, val_all_predictions, val_all_targets = model.evaluate(val_loader, utils.device, return_predictions=True)
    test_avg_loss, test_all_predictions, test_all_targets = model.evaluate(test_loader, utils.device, return_predictions=True)

    
    # # Save predictions to CSV for later plotting
    out_dir = Path("predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = os.path.join(out_dir, f"{args.module}_predictions.csv")
    
    result_train_df, result_val_df, result_test_df, result_df = utils.save_predictions(train_all_predictions, train_all_targets, \
                                        val_all_predictions, val_all_targets, \
                                        test_all_predictions, test_all_targets, \
                                        predictions_path)
                                    
    # Create plots and log to wandb
    logger.log_predictions(result_train_df, result_val_df, result_test_df)
    
    print("\nTraining completed successfully!")
    wandb.finish()

if __name__ == "__main__":
    main()
