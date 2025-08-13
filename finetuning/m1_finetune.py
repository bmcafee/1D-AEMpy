import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


import sys
sys.path.append("..") 
from utility import MLP, DataGenerator, rmse, l2_error


# CUDA support 
if torch.cuda.is_available():
    device = torch.device('cuda:5')
else:
    device = torch.device('cpu')
    
print(device)
device = torch.device('cpu')



def get_rollout_predictions(m1_model, m2_model, m3_model, m4_model, loader, plot = True):    
    m1_model.eval()
    m2_model.eval()
    m3_model.eval()
    m4_model.eval()

    y_ = []
    y_obs_ = []
    pred_ = []
        
    rmse_models = np.zeros((len(loader), 5))
    for ix, x in enumerate(iter(loader)):
        x = x.to(device).float()
        
        m1_input = x[:, m1_input_column_ix]
            
        #model 1
        m1_pred = m1_model(m1_input)
        # breakpoint()
        if plot:
            m1_y_true = x[:, m1_output_column_ix[0]] * torch.tensor(train_std[m1_output_column_ix[0]]).to(device) + torch.tensor(train_mean[m1_output_column_ix[0]]).to(device)
            m1_y_pred = m1_pred * torch.tensor(train_std[m1_output_column_ix[0]]).to(device) + torch.tensor(train_mean[m1_output_column_ix[0]]).to(device)
            rmse_models[ix, 0] = rmse(m1_y_true.squeeze(), m1_y_pred.squeeze())
            print("RMSE of after m1", rmse(m1_y_true.flatten(), m1_y_pred.flatten()))

        #model 2
        m2_input = torch.cat([x[:, m2_input_iso_ix], m1_pred], dim=-1)
        m2_pred = m2_model(m2_input)
        # breakpoint()
        if plot:
            m2_y_true = x[:, m2_output_column_ix] * torch.tensor(train_std[m2_output_column_ix]).to(device) + torch.tensor(train_mean[m2_output_column_ix]).to(device)
            m2_y_pred = m2_pred * torch.tensor(train_std[m2_output_column_ix]).to(device) + torch.tensor(train_mean[m2_output_column_ix]).to(device)
            rmse_models[ix, 1] = rmse(m2_y_true.squeeze(), m2_y_pred.squeeze())
            print("RMSE of after m2", rmse(m2_y_true.flatten(), m2_y_pred.flatten()))

        #model 3
        m2_output_shared_ix = [ix for ix in m2_output_column_ix if ix in m3_input_column_ix]
        relative_shared_ix = [m2_output_column_ix.index(ix) for ix in m2_output_shared_ix]
        m2_pred_selected = m2_pred[:, relative_shared_ix]
        m3_input = torch.cat([x[:, m3_input_iso_ix], m2_pred_selected], dim=-1)
        m3_pred = m3_model(m3_input)
        # breakpoint()    
        if plot:
            m3_y_true = x[:, m3_output_column_ix] * torch.tensor(train_std[m3_output_column_ix]).to(device) + torch.tensor(train_mean[m3_output_column_ix]).to(device)
            m3_y_pred = m3_pred * torch.tensor(train_std[m3_output_column_ix]).to(device) + torch.tensor(train_mean[m3_output_column_ix]).to(device)
            rmse_models[ix, 3] = rmse(m3_y_true.squeeze(), m3_y_pred.squeeze())
            print("RMSE of after m3", rmse(m3_y_true.flatten(), m3_y_pred.flatten()))

        # breakpoint()
            
        #model 4
        m3_output_shared_ix = [ix for ix in m3_output_column_ix if ix in m4_input_column_ix]
        relative_shared_ix = [m3_output_column_ix.index(ix) for ix in m3_output_shared_ix]
        m3_pred_selected = m3_pred[:, relative_shared_ix]
        m4_input = torch.cat([x[:, m4_input_iso_ix], m3_pred_selected], dim=-1)
        m4_pred = m4_model(m4_input)
        # breakpoint()

        mean = torch.tensor(train_mean[m4_output_column_ix]).to(device)
        std = torch.tensor(train_std[m4_output_column_ix]).to(device)
        if plot:
            m4_y_true = x[:, m4_output_column_ix] * std + mean
            m4_y_pred = m4_pred * std + mean
            rmse_models[ix, 4] = rmse(m4_y_true.squeeze(), m4_y_pred.squeeze())
            print("RMSE of after m4", rmse(m4_y_true.flatten(), m4_y_pred.flatten()))

        # store the predictions
        y_true = x[:, m4_output_column_ix] * std + mean
        y_obs = x[:, obs_columns_ix] * train_std[obs_columns_ix] + train_mean[obs_columns_ix]
        pred = m4_pred * std + mean
        
        y_.append(y_true)
        y_obs_.append(y_obs)
        pred_.append(pred)

    y_ = torch.cat(y_, dim=0)
    y_obs_ = torch.cat(y_obs_, dim=0)
    pred_ = torch.cat(pred_, dim=0) 

    return pred_, y_, y_obs_, rmse_models

# Load data
folder_path = '/raid/sepideh/Project_MCL/1D-AEMpy-UW-metabolism-BM'
file_path = os.path.join(folder_path, "all_data_lake_modeling_mixed.csv")

df = pd.read_csv(file_path)

time = df['datetime']
df = df.drop(columns=['datetime'])


# create new variables
df['doc_diff04'] = df['docr_diff04'] + df['docl_diff04']
df['poc_diff04'] = df['pocr_diff04'] + df['pocl_diff04']


# Split data
training_frac = 0.60
val_frac = 0.2

depth_steps = 50
number_days = len(df)//depth_steps

# Calculate number of observations for each set
n_obs_train = int(number_days * training_frac) * depth_steps
n_obs_val = int(number_days * val_frac) * depth_steps

print(f"Number of days total: {number_days}")

train_df = df.iloc[:n_obs_train]
val_df = df.iloc[n_obs_train:n_obs_train + n_obs_val]
test_df = df.iloc[n_obs_train + n_obs_val:]

print(f"Number of training points: {train_df.shape}")
print(f"Number of validation points: {val_df.shape}")
print(f"Number of test points: {test_df.shape}")

# Specify input and output columns

input_binary = ['ice_final06'] # remain unchanged

features_to_normalize = list(set(df.columns) - set(input_binary))

train_features_to_normalize = train_df[features_to_normalize].values
val_data_features_to_normalize = val_df[features_to_normalize].values
test_features_to_normalize = test_df[features_to_normalize].values

# Initialize StandardScaler and fit on training features
scaler = StandardScaler()
scaler.fit(train_features_to_normalize)

# Transform training, test, and validation features using the fitted scaler
train_df[features_to_normalize] = scaler.transform(train_features_to_normalize)
val_df[features_to_normalize] = scaler.transform(val_data_features_to_normalize)
test_df[features_to_normalize] = scaler.transform(test_features_to_normalize)
# --------------------------------------------------------------------------
#keeping track of the mean and standard deviations
train_mean = scaler.mean_
train_std = scaler.scale_

# Get index of the binary column
ice_ix = df.columns.get_loc("ice_final06")

# Insert placeholder mean/std at the correct index
train_mean_full = np.insert(scaler.mean_, ice_ix, 0.0)  # mean for binary is 0
train_std_full = np.insert(scaler.scale_, ice_ix, 1.0)  # std for binary is 1 (no scaling)

train_mean = train_mean_full
train_std = train_std_full

# --------------------------------------------------------------------------
# define obs columns
obs_columns = ['do_obs', 'doc_obs', 'poc_obs']

obs_columns_ix = [df.columns.get_loc(column) for column in obs_columns]

# --------------------------------------------------------------------------
# m1: Input and Output
m1_input_columns = ['depth', 
                    'day_of_year_list', 'time_of_day_list', # time features
                    'temp_initial00', 'do_initial00', # 2D var
                    'area_input', 'volume_input', # depth specific var
                    'Air_Temperature_celsius', 'Ten_Meter_Elevation_Wind_Speed_meterPerSecond', # Meteorological var, time specific var
                    'ice_final06']
m1_output_columns = ['do_ax01']

# Get index positions for columns
m1_input_column_ix = [df.columns.get_loc(col) for col in m1_input_columns]
m1_output_column_ix = [df.columns.get_loc(col) for col in m1_output_columns]


# Exclude the inputs which are output of previous model
# pass

# Model config
m1_PATH = f"../saved_models/pretrain/m1.pth"
m1_layers = [len(m1_input_columns), 32, 32, len(m1_output_columns)]

# Load model
m1_model = MLP(m1_layers, activation="gelu")
m1_checkpoint = torch.load(m1_PATH, map_location=torch.device('cpu'))
m1_model.load_state_dict(m1_checkpoint)
m1_model = m1_model.to(device)

# --------------------------------------------------------------------------
# m2: Input and Output
m2_input_columns = ['depth', 
                    'day_of_year_list', 'time_of_day_list', # time features
                    'temp_initial00', 'do_ax01', 'docr_initial00', 'docl_initial00', 'pocr_initial00', 'pocl_initial00',
                    'area_input', 'volume_input', 'ice_final06',
                    'Air_Temperature_celsius', 'Cloud_Cover', 'ea', 'Shortwave_Radiation_Downwelling_wattPerMeterSquared' , 'Longwave_Radiation_Downwelling_wattPerMeterSquared',
                    'tp_initial']
m2_output_columns = ['do_bc02', 'docr_bc02', 'docl_bc02', 'pocr_bc02', 'pocl_bc02', 'npp_bc02']

# Get index positions for columns
m2_input_column_ix = [df.columns.get_loc(col) for col in m2_input_columns]
m2_output_column_ix = [df.columns.get_loc(col) for col in m2_output_columns]

# Exclude the inputs which are output of previous model
m2_input_iso_ix = [ix for ix in m2_input_column_ix if ix not in m1_output_column_ix]

# Model config
m2_PATH = f"../saved_models/pretrain/m2.pth"
m2_layers = [len(m2_input_columns), 32, 32, len(m2_output_columns)]

# Load model
m2_model = MLP(m2_layers, activation="gelu")
m2_checkpoint = torch.load(m2_PATH, map_location=torch.device('cpu'))
m2_model.load_state_dict(m2_checkpoint)
m2_model = m2_model.to(device)

# --------------------------------------------------------------------------
# m3: Input and Output
m3_input_columns = ['depth', 
                    'day_of_year_list', 'time_of_day_list', # time features
                    
                    'temp_initial00', 'do_bc02', 'docl_bc02', 'docr_bc02', 'pocl_bc02', 'pocr_bc02', 
                    'area_input', 'volume_input',]
m3_output_columns = ['do_pd03', 'docr_pd03','docl_pd03','pocr_pd03','pocl_pd03', 'docr_resp_pd03', 'docl_resp_pd03','poc_resp_pd03']

# Get index positions for columns
m3_input_column_ix = [df.columns.get_loc(col) for col in m3_input_columns]
m3_output_column_ix = [df.columns.get_loc(col) for col in m3_output_columns]


# Exclude the inputs which are output of previous model
m3_input_iso_ix = [ix for ix in m3_input_column_ix if ix not in m2_output_column_ix]


# Model config
m3_PATH = f"../saved_models/pretrain/m3.pth"
m3_layers = [len(m3_input_columns), 32, 32, len(m3_output_columns)]

# Load model
m3_model = MLP(m3_layers, activation="gelu")
m3_checkpoint = torch.load(m3_PATH, map_location=torch.device('cpu'))
m3_model.load_state_dict(m3_checkpoint)
m3_model = m3_model.to(device)

# --------------------------------------------------------------------------
# m4: Input and Output
m4_input_columns = ['depth', 
                    'day_of_year_list', 'time_of_day_list', # time features
                    
                    # inputs
                    'temp_initial00', 'do_pd03', 'docl_pd03', 'docr_pd03', 'pocl_pd03', 'pocr_pd03', 'kz_initial00', 
                    'Ten_Meter_Elevation_Wind_Speed_meterPerSecond', 
                    'area_input', 'volume_input',]
m4_output_columns = ['do_diff04', 'poc_diff04', 'doc_diff04']

# Get index positions for columns
m4_input_column_ix = [df.columns.get_loc(col) for col in m4_input_columns]
m4_output_column_ix = [df.columns.get_loc(col) for col in m4_output_columns]


# Exclude the inputs which are output of previous model
m4_input_iso_ix = [ix for ix in m4_input_column_ix if ix not in m3_output_column_ix]

# Model config
m4_PATH = f"../saved_models/pretrain/m4.pth"
m4_layers = [len(m4_input_columns), 32, 32, len(m4_output_columns)]

# Load model
m4_model = MLP(m4_layers, activation="gelu")
m4_checkpoint = torch.load(m4_PATH, map_location=torch.device('cpu'))
m4_model.load_state_dict(m4_checkpoint)
m4_model = m4_model.to(device)


# --------------------------------------------------------------------------



    
train_array = train_df.values
test_array = test_df.values
val_array = val_df.values

# Create data set
batch_size = 1024
train_dataset = DataGenerator(train_df)
val_dataset = DataGenerator(val_df)
test_dataset = DataGenerator(test_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          shuffle=False)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)


print(train_df.shape)
print(val_df.shape)

# --------------------------------------------------------------------------
train_y_pred, train_y_true, train_y_obs, train_rmse_models = get_rollout_predictions(m1_model, m2_model, m3_model, m4_model, train_loader, plot = True)

train_rmse = rmse(train_y_pred.flatten(), train_y_true.flatten())
train_rmse_obs = rmse(train_y_pred.flatten(), train_y_obs.flatten())
train_l2 = l2_error(train_y_pred.flatten(), train_y_true.flatten())

print(f"Train RMSE Simulated: {train_rmse}")
print(f"Train RMSE Observed Temp: {train_rmse_obs}")
print(f"Train L2 Error: {train_l2}")
print(f"The RMSEs after each modelling stage: {train_rmse_models.mean(axis=0)}")