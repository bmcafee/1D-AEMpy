
import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Logger:
    def __init__(self, utils_instance):
        self.utils = utils_instance

    def log_data_info(self, df_m):
        # Put static run info in config/summary (no charts per step)
        wandb.config.update({
            'input_columns': self.utils.model_config['input_columns'],
            'output_columns': self.utils.model_config['output_columns'],
        }, allow_val_change=True)
        wandb.run.summary['data_shape'] = tuple(df_m.shape)
        wandb.run.summary['total_samples'] = int(len(df_m))

    def log_model_info(self, model, input_size, output_size):
        # Static info → summary (no time-series chart)
        wandb.run.summary['model_architecture'] = str(model)
        wandb.run.summary['input_size'] = int(input_size)
        wandb.run.summary['output_size'] = int(output_size)
        wandb.run.summary['total_parameters'] = sum(p.numel() for p in model.parameters())
        wandb.run.summary['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def log_training_step(self, epoch, train_loss, val_loss, learning_rate):
        # These are the only things you typically want as time series
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': learning_rate
        }, step=epoch)

    def log_training_curves(self, train_loss, val_loss, test_loss=None):
        # Keep the figure; REMOVE logging raw lists (that created extra plots)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_loss, label='Train Loss', linewidth=2)
        ax.plot(val_loss, label='Validation Loss', linewidth=2)
        if test_loss is not None:
            ax.plot(test_loss, label='Test Loss', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training, Validation, and Test Loss')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')
        wandb.log({"training_curves": wandb.Image(fig)})
        plt.close(fig)

        # If you still want to save the raw arrays, store once in summary (no charts)
        wandb.run.summary['train_loss_history'] = list(map(float, train_loss))
        wandb.run.summary['val_loss_history']   = list(map(float, val_loss))
        if test_loss is not None:
            wandb.run.summary['test_loss_history'] = list(map(float, test_loss))

    def log_final_metrics(self, train_loss, val_loss, train_metrics, val_metrics, test_metrics, output_columns,
    val_metrics_unnormalized, train_metrics_unnormalized, test_metrics_unnormalized):
        # Final scalars are fine in summary to avoid extra charts
        s = wandb.run.summary
        s['final_train_loss'] = float(train_loss[-1])
        s['final_val_loss']   = float(val_loss[-1])
        
        # s['best_val_loss']    = float(min(val_loss))
        # s['best_train_loss']  = float(min(train_loss))

        for i, col in enumerate(output_columns):
            s[f'val_rmse_{col}']   = float(val_metrics[0][i])
            s[f'train_rmse_{col}'] = float(train_metrics[0][i])
            s[f'test_rmse_{col}']  = float(test_metrics[0][i])

        s['val_overall_rmse']   = float(val_metrics[2])
        s['train_overall_rmse'] = float(train_metrics[2])
        s['test_overall_rmse']  = float(test_metrics[2])

        s['val_overall_rmse_unnorm']   = float(val_metrics_unnormalized[2])
        s['train_overall_rmse_unnorm'] = float(train_metrics_unnormalized[2])
        s['test_overall_rmse_unnorm']  = float(test_metrics_unnormalized[2])

    def log_predictions(self, result_train_df, result_val_df, result_test_df):
        
        # Filter for one specific depth and use all samples
        target_depth = 0  # Surface depth
        
        # Filter each split for the target depth (use all samples at this depth)
        train_sample = result_train_df[result_train_df['depth'] == target_depth].copy()
        val_sample = result_val_df[result_val_df['depth'] == target_depth].copy()
        test_sample = result_test_df[result_test_df['depth'] == target_depth].copy()
        
        print(f"Plotting ALL samples at depth {target_depth}m - Train: {len(train_sample)}, Val: {len(val_sample)}, Test: {len(test_sample)} samples")
        
        # Convert datetime columns to datetime if they're not already
        for df in [train_sample, val_sample, test_sample]:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Get output columns from config
        output_cols = self.utils.output_columns
        
        # Create subplots: n_outputs rows x 3 columns (train/val/test)
        n_outputs = len(output_cols)
        fig, axes = plt.subplots(n_outputs, 3, figsize=(20, 5*n_outputs))
        if n_outputs == 1:
            axes = axes.reshape(1, -1)  # Ensure 2D array even for single output
        
        split_names = ['train', 'val', 'test']
        split_dfs = [train_sample, val_sample, test_sample]
        
        for i, output_col in enumerate(output_cols):
            for j, (split_name, split_df) in enumerate(zip(split_names, split_dfs)):
                ax = axes[i, j]
                
                if len(split_df) > 0:
                    pred_col = f'{output_col}_pred'
                    true_col = f'{output_col}_true'
                    # Sort by datetime to ensure proper time series plotting
                    split_df_sorted = split_df.sort_values('datetime')
                    
                    # Plot ground truth as thick solid line in dark color
                    ax.plot(split_df_sorted['datetime'], split_df_sorted[true_col], linewidth=3, alpha=0.6,label='Ground Truth', linestyle='-')
                    
                    # Plot predictions as thinner line in contrasting bright color
                    ax.plot(split_df_sorted['datetime'], split_df_sorted[pred_col], linewidth=2, label='Prediction', linestyle='-')
                
                ax.set_xlabel('Time')
                ax.set_ylabel(f'{output_col}')
                ax.set_title(f'{output_col} - {split_name.title()} Set')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Rotate x-axis labels for better readability
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({f"prediction_time_series at depth {target_depth}": wandb.Image(fig)})
        plt.close(fig)


