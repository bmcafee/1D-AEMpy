#!/usr/bin/env python3
"""
Model definitions for pretraining scripts
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

import pandas as pd
import numpy as np


def rmse(self, true, pred):
    """Calculate RMSE"""
    return (((true - pred) ** 2).mean() ** 0.5).detach().cpu().numpy()

def l2_error(self, true, pred):
    """Calculate L2 error"""
    return np.linalg.norm(pred.detach().cpu().numpy() - true.detach().cpu().numpy()) / np.linalg.norm(true.detach().cpu().numpy())



class MLP(torch.nn.Module):
    """Multi-Layer Perceptron model"""
    
    def __init__(self, layers, activation="relu", init="xavier", dropout_rate=0.0):
        super(MLP, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        self.dropout_rate = dropout_rate
        
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        else:
            raise ValueError("Unspecified activation type")
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))
            # Add dropout after activation (except for the last layer)
            if self.dropout_rate > 0.0:
                layer_list.append(('dropout_%d' % i, torch.nn.Dropout(p=self.dropout_rate)))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

        if init == "xavier":
            self.xavier_init_weights()
        elif init == "kaiming":
            self.kaiming_init_weights()
    
    def xavier_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Xavier Initialization..")
            for m in self.layers.modules():
                if hasattr(m, 'weight'):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.fill_(0.0)

    def kaiming_init_weights(self):
        with torch.no_grad():
            print("Initializing Network with Kaiming Initialization..")
            for m in self.layers.modules():
                if hasattr(m, 'weight'):
                    nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.fill_(0.0)
                        
    def forward(self, x):
        out = self.layers(x)
        return out    

    def run_training(self, train_loader, val_loader, test_loader, args, device, logger):
        """Train the model"""
        # Setup training
        print("Setting up training...")
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_steps, gamma=args.decay_rate)
        criterion = torch.nn.MSELoss()
        
        train_loss = []
        val_loss = []
        test_loss = []
        
        # Progress bar for epochs
        epoch_pbar = tqdm(range(args.num_epochs), desc="Training", leave=True)
        
        try:
            for epoch in epoch_pbar:
                # Training
                self.train()
                train_loss_epoch = 0
                
                # Training step
                for x, y in train_loader:
                    x, y = x.to(device).float(), y.to(device).float()
                    optimizer.zero_grad()
                    pred = self(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
                    train_loss_epoch += loss.detach().item()
                
                lr_scheduler.step()
                
                # Compute losses for this epoch
                train_loss_avg = train_loss_epoch / len(train_loader)
                train_loss.append(train_loss_avg)
                
                # Validation and test loss (every epoch)
                val_loss_avg = self.evaluate(val_loader, device, return_predictions=False)
                test_loss_avg = self.evaluate(test_loader, device, return_predictions=False)
                val_loss.append(val_loss_avg)
                test_loss.append(test_loss_avg)
                
                # Log to wandb
                logger.log_training_step(epoch, train_loss_avg, val_loss_avg, optimizer.param_groups[0]['lr'])
                
                # Print epoch summary
                print(f"\n Epoch {epoch}: Train={train_loss_avg:.6f}, Val={val_loss_avg:.6f}, Test={test_loss_avg:.6f}")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user!")
            epoch_pbar.close()
        except Exception as e:
            print(f"\nTraining error occurred: {e}")
            epoch_pbar.close()
            raise
        finally:
            epoch_pbar.close()
        
        return train_loss, val_loss, test_loss
    
    
    def evaluate(self, data_loader, device, return_predictions=True):
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        criterion = torch.nn.MSELoss()

        with torch.no_grad():
            self.eval()
            for x, y in data_loader:
                x, y = x.to(device).float(), y.to(device).float()
                pred = self(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
                
                if return_predictions:
                    all_predictions.append(pred)
                    all_targets.append(y)
        
        avg_loss = total_loss / len(data_loader)
        
        if return_predictions:
            # Concatenate all batches
            all_predictions = torch.cat(all_predictions, dim=0).cpu()
            all_targets = torch.cat(all_targets, dim=0).cpu()
            return avg_loss, all_predictions, all_targets
        else:
            return avg_loss
