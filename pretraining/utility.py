#!/usr/bin/env python3
"""
Utility classes and functions for pretraining scripts
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# Suppress warnings
warnings.filterwarnings("ignore")
    

class Utils:
    """Utility class for pretraining operations"""
    
    def __init__(self, config_file, cuda_device, data_folder, data_file, model_name):
        """Initialize Utils with common parameters"""
        self.config = self.load_config(config_file)
        self.model_config = self.config[model_name]
        self.device = self.setup_device(cuda_device)
        self.data_folder = data_folder
        self.data_file = data_file
        self.output_columns = self.model_config['output_columns']
        self.input_columns = self.model_config['input_columns']
        self.input_binary = self.model_config['input_binary']
        self.depth_steps = self.model_config['depth_steps']
        self.training_frac = self.model_config['training_frac']
        self.val_frac = self.model_config['val_frac']
        
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    
    def setup_device(self, cuda_device):
        """Setup device (GPU/CPU) for training"""
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{cuda_device}')
            print(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
        return device
    
    def set_random_seed(self, seed=42):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Random seed set to {seed} for reproducibility")
    

    def plot_training_curves(self, train_loss, val_loss):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(8, 6))
        plt.plot(train_loss, label="Train", linewidth=2.5)
        plt.plot(val_loss, label="Validation", linewidth=2.5)
        plt.grid("on", alpha=0.2)
        plt.legend(fontsize=18)
        plt.yscale("log")
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("Loss", fontsize=18)
        plt.title("Training and Validation Loss")
        plt.show()
    
    def save_model(self, model, save_path):
        """Save the trained model"""
        print(f"Saving model to {save_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(model.state_dict(), save_path)
        print("Model saved successfully!")
    
    
    def split_data(self, df_m):
        """Split data into train/validation/test sets"""
        print("Splitting data...")
        
        number_days = len(df_m) // self.model_config['depth_steps']
        n_obs_train = int(number_days * self.model_config['training_frac']) * self.model_config['depth_steps']
        n_obs_val = int(number_days * self.model_config['val_frac']) * self.model_config['depth_steps']
        
        train_df = df_m.iloc[:n_obs_train]
        val_df = df_m.iloc[n_obs_train:n_obs_train + n_obs_val]
        test_df = df_m.iloc[n_obs_train + n_obs_val:]

        self.n_train = len(train_df)
        self.n_val   = len(val_df)
        self.n_test  = len(test_df)
        
        
        print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        return train_df, val_df, test_df
    

    def store_datetime_depth_column(self, df, datetime_column='datetime', depth_column='depth'):
        """Store datetime and depth columns for later use in saving predictions"""
        self.datetime_column = df[datetime_column].copy()
        self.depth_column = df[depth_column].copy()

    def normalize_features(self, train_df, val_df, test_df):
        """Normalize features using StandardScaler"""
        print("Normalizing features...")
        
        all_columns = self.model_config['input_columns'] + self.model_config['output_columns']
        features_to_normalize = [c for c in train_df.columns if c in all_columns and c not in self.input_binary]
        scaler = StandardScaler()
        scaler.fit(train_df[features_to_normalize].values)
        
        train_df[features_to_normalize] = scaler.transform(train_df[features_to_normalize].values)
        val_df[features_to_normalize] = scaler.transform(val_df[features_to_normalize].values)
        test_df[features_to_normalize] = scaler.transform(test_df[features_to_normalize].values)
        self.scaler = scaler
        
        # Store the features_to_normalize list to use for denormalization
        self.features_to_normalize = features_to_normalize
        
        # Create mapping from output columns to their positions in the normalized features array
        self.output_indices_in_scaler = []
        for col in self.output_columns:
            if col in features_to_normalize:
                self.output_indices_in_scaler.append(features_to_normalize.index(col))        
        return train_df, val_df, test_df, scaler


    
    def save_predictions(self, train_all_predictions, train_all_targets, \
        val_all_predictions, val_all_targets, \
        test_all_predictions, test_all_targets, \
        predictions_path):

        # Convert all splits to DataFrames
        result_train_df = self.to_dataframe(train_all_predictions, train_all_targets, split_name="train")
        result_val_df   = self.to_dataframe(val_all_predictions, val_all_targets, split_name="val")
        result_test_df  = self.to_dataframe(test_all_predictions, test_all_targets, split_name="test")

        # Concatenate all and save
        result_df = pd.concat([result_train_df, result_val_df, result_test_df], ignore_index=True)
        result_df.to_csv(predictions_path, index=False)

        return result_train_df, result_val_df, result_test_df, result_df
    
    def to_dataframe(self, preds, targets, split_name):
        """
        Convert predictions and targets to a DataFrame with proper column naming.
        Denormalizes the values before saving.
        """
        # Denormalize predictions and targets
        # preds_denorm = self.denorm_outputs(preds)
        # targets_denorm = self.denorm_outputs(targets)
        
        # Convert tensors to numpy arrays
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Get the correct slice of datetime and depth for this split
        if split_name == 'train':
            start_idx, end_idx = 0, self.n_train
        elif split_name == 'val':
            start_idx, end_idx = self.n_train, self.n_train + self.n_val
        elif split_name == 'test':
            start_idx, end_idx = self.n_train + self.n_val, self.n_train + self.n_val + self.n_test
        else:
            raise ValueError(f"Unknown split_name: {split_name}")
        
        # Create DataFrame dictionary
        df_dict = {
            'split': [split_name] * len(preds_np),
            'datetime': self.datetime_column.iloc[start_idx:end_idx],
            'depth': self.depth_column.iloc[start_idx:end_idx]
        }
        # Plot start date and end date of each split
        print(f"Start date of {split_name} split: {self.datetime_column.iloc[start_idx]}")
        print(f"End date of {split_name} split: {self.datetime_column.iloc[end_idx-1]}")
    
        # Add predictions and targets
        for i, col in enumerate(self.output_columns):
            df_dict[f'{col}_pred'] = preds_np[:, i]
            df_dict[f'{col}_true'] = targets_np[:, i]

        return pd.DataFrame(df_dict)

    def data_loaders(self, train_df, val_df, test_df, batch_size):
        """Prepare data loaders for training"""
        print("Preparing data loaders...")
        
        input_columns = self.model_config['input_columns']
        output_columns = self.model_config['output_columns']
        
        self.input_column_ix = [train_df.columns.get_loc(column) for column in input_columns]
        self.output_column_ix = [train_df.columns.get_loc(column) for column in output_columns]
        
        # Create datasets
        train_array, val_array, test_array = train_df.values, val_df.values, test_df.values
        
        X_train = train_array[:, self.input_column_ix]
        X_val = val_array[:, self.input_column_ix]
        X_test = test_array[:, self.input_column_ix]
        
        y_train = train_array[:, self.output_column_ix]
        y_val = val_array[:, self.output_column_ix]
        y_test = test_array[:, self.output_column_ix]
        
        # Create data loaders
        train_dataset = DataGenerator(X_train, y_train)
        val_dataset = DataGenerator(X_val, y_val)
        test_dataset = DataGenerator(X_test, y_test)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")
        
        return train_loader, val_loader, test_loader

    def denorm_outputs(self, y_norm):
        # y_norm: (batch, n_out) tensor on any device
        # Get the correct indices from the scaler for output columns
        means = torch.tensor(self.scaler.mean_[self.output_indices_in_scaler], device=y_norm.device, dtype=y_norm.dtype)
        scales = torch.tensor(self.scaler.scale_[self.output_indices_in_scaler], device=y_norm.device, dtype=y_norm.dtype)
        return y_norm * scales + means

class DataGenerator(torch.utils.data.Dataset):
    """Custom dataset for training"""
    
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    
    def __len__(self):
        return len(self.X)

