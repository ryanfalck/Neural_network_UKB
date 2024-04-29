#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:58:26 2024

@author: ryan
"""
# Data wrangling
import pandas as pd
import numpy as np
import sklearn
import os

# Data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import snips as snp  # my snippets

#Machine Learning Packages from Scikit-learn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, learning_curve, check_cv
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import preprocessing

#Machine Learning Packages from Pytorch
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import optuna
from optuna.exceptions import TrialPruned
import optuna.visualization as vis

os.getcwd()
os.chdir('/Users/ryan/Desktop/UBC-Postdoctoral Fellowship/UK Biobank Project/Data')
os.listdir()

#Load CSV File by Chunks
sine_cog = pd.read_csv('Sine.csv')

""" 1) Pipieline Set-up """

#Set Seed    
seed = 3 
torch.manual_seed(seed)

#Specify levels of ordinal variables
educational_cat = ['Prefer not to answer', 'None of the above', 'Other professional qualifications', 
                          'NVQ/HND/HNC', 'CSEs', 'O levels/GCSEs', 'A/AS levels', 'University degree']

income_cat = ['Prefer not to answer', 'Do not know', '<18,000 GBP', '18,000 to 30,999 GBP', '31,000 to 51,999 GBP',
                     '52,000 to 100,000 GBP', '>100,000 GBP']

smoking_cat = ['Prefer not to answer', 'Current Smoker', 'Previous Smoker', 'Never']

alcohol_cat = ['Prefer not to answer', 'Never', 'Special occasions only', '1-3x/month', '1-2x/week', '3-4x/week', 'Daily or almost daily']

    
#Initiate ordinal encoders
oe_education =  OrdinalEncoder(categories = [educational_cat])
oe_income =  OrdinalEncoder(categories = [income_cat])
oe_smoking =  OrdinalEncoder(categories = [smoking_cat])
oe_alcohol =  OrdinalEncoder(categories = [alcohol_cat])


#Initiate One Hot Encoder
OneHot = OneHotEncoder(sparse=False)


#Specify numeric features
numeric_features_sine = ['Age at baseline', 'Townsend Deprivation Index', 'BMI',
                    'Amplitude', 'Frequency', 'Phase', 'Offset']

outcome_features = ['Trails A Time', 'Trails B Time', 'Trails', 'DSST Total Score']


#Initiate numeric transformer with standard scaler
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])


# Get predictor variables and target variables from data
X_sine = sine_cog.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13]]
DSST_Y = sine_cog.iloc[:,[14]]


#Make column transformer which consists of OneHotEncoder and Ordinal 
preprocessor_sine = make_column_transformer(
    (OneHot, ['Sex', 'Ethnicity']),
    (oe_education, ['Education']),
    (oe_income, ['Household Income']),
    (oe_smoking, ['Smoking Status']),
    (oe_alcohol, ['Alcohol Intake']),
    (numeric_transformer, ['Baseline Age', 'Townsend Deprivation Index', 'BMI',
                        'Amplitude', 'Frequency', 'Phase', 'Offset']))




#Apply column transformer to predictor variables
X_sine_2 = preprocessor_sine.fit_transform(X_sine)


#Outcome variables are fine; reshape to an array for numpy
DSST_Y2 = DSST_Y.to_numpy()
 


#Train/validation/test splits for DSST from Sine Wave
X_sine_DSST_train, X_sine_DSST_test, Y_sine_DSST_train, Y_sine_DSST_test = train_test_split(X_sine_2, DSST_Y2,
                                                                                test_size = 0.3, random_state =2)

X_sine_DSST_test, X_sine_DSST_val, Y_sine_DSST_test, Y_sine_DSST_val = train_test_split(X_sine_DSST_test, Y_sine_DSST_test,
                                                                                test_size = 0.5, random_state =2)


#Data Set-up for DSST
X_sine_DSST_train.shape, X_sine_DSST_val.shape, Y_sine_DSST_train.shape, Y_sine_DSST_val.shape
Y_sine_DSST_train_1 = np.asarray(np.squeeze(Y_sine_DSST_train))   
Y_sine_DSST_val_1 = np.asarray(np.squeeze(Y_sine_DSST_val))


# Data Loader Function (Note: this is the same for all models)
from torch.utils.data import Dataset, DataLoader
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)  # Ensure target tensor has shape [num_samples, 1]
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):      
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len
    
# Data Loaders for DSST
train_sine_DSST_data = Data(X_sine_DSST_train, Y_sine_DSST_train_1)
train_sine_DSST_dataloader = DataLoader(dataset=train_sine_DSST_data, batch_size=20000, shuffle=True)

val_sine_DSST_data = Data(X_sine_DSST_val, Y_sine_DSST_val_1)
val_sine_DSST_dataloader = DataLoader(dataset=val_sine_DSST_data, batch_size=20000, shuffle=True)

# Convert the numpy arrays to PyTorch tensors for DSST
X_sine_train_DSST_tensor = torch.tensor(X_sine_DSST_train, dtype=torch.float32)
Y_sine_train_DSST_tensor = torch.tensor(Y_sine_DSST_train_1, dtype=torch.long)
Y_sine_train_DSST_tensor = Y_sine_train_DSST_tensor.unsqueeze(1)
X_sine_val_DSST_tensor = torch.tensor(X_sine_DSST_val, dtype=torch.float32)
Y_sine_val_DSST_tensor = torch.tensor(Y_sine_DSST_val_1, dtype=torch.long)
Y_sine_val_DSST_tensor = Y_sine_val_DSST_tensor.unsqueeze(1)


# Define the neural network model (Note: this is the same for all models)
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dims):
        super(NeuralNetwork, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)  # Reshape output tensor
        return out.squeeze()  # Remove the extra dimension


def objective(trial):
    # Define the hyperparameters search space
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 10)
    hidden_layer_dims = []
    for i in range(num_hidden_layers):
        hidden_dim = trial.suggest_categorical(f"hidden_dim_{i}", [1,10,100,1000])
        hidden_layer_dims.append(hidden_dim)
    
    # Define the model
    input_dim = 20
    output_dim = 1
    model = NeuralNetwork(input_dim, output_dim, hidden_layer_dims)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-7)
    
    DSST_sine_best_loss = float("inf")
    patience = 5
    early_stopping_counter = 0
    
    # Define lists to store training curve data
    DSST_sine_train_losses = []
    DSST_sine_val_losses = []
    
    X_sine_DSST_val_tensor = torch.tensor(X_sine_DSST_val, dtype=torch.float32).view(-1, input_dim)
    Y_sine_DSST_val_tensor = torch.tensor(Y_sine_DSST_val_1, dtype=torch.long)

    # Train the model
    model.train()
    for epoch in range(4000):
        DSST_sine_running_loss = 0.0
        for inputs, targets in train_sine_DSST_dataloader:
            optimizer.zero_grad()
            DSST_sine_outputs = model(inputs)
            DSST_sine_loss = criterion(DSST_sine_outputs.squeeze(), targets.squeeze())
            DSST_sine_loss.backward()
            optimizer.step()
            DSST_sine_running_loss += DSST_sine_loss.item()
        
        DSST_epoch_loss = DSST_sine_running_loss / len(train_sine_DSST_dataloader)
        DSST_sine_train_losses.append(DSST_epoch_loss)
        
        # Early stopping based on validation loss
        with torch.no_grad():
            model.eval()
            DSST_sine_predictions = model(X_sine_val_DSST_tensor)
            DSST_sine_val_loss = criterion(DSST_sine_predictions, Y_sine_val_DSST_tensor)
            DSST_sine_val_losses.append(DSST_sine_val_loss.item())
            
            if DSST_sine_val_loss < DSST_sine_best_loss:
                DSST_sine_best_loss = DSST_sine_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= patience:
                break
     
    # Save the training curve data in the trial user attributes
    trial.set_user_attr("train_losses", DSST_sine_train_losses)
    trial.set_user_attr("val_losses", DSST_sine_val_losses)  
        
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        DSST_sine_predictions = model(X_sine_val_DSST_tensor)
        DSST_sine_mse = criterion(DSST_sine_predictions.squeeze(), Y_sine_val_DSST_tensor)
    
    return DSST_sine_mse.item()

# Create an Optuna study with a pruner
DSST_sine_study1 = optuna.create_study(direction="minimize")

# Generate all possible combinations of hyperparameters
DSST_sine_grid_search = {
    "num_hidden_layers": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "hidden_dim": [1,10,100,1000]
}

for num_hidden_layers in DSST_sine_grid_search["num_hidden_layers"]:
    for hidden_dim in DSST_sine_grid_search["hidden_dim"]:
        DSST_sine_study1.enqueue_trial({
            "num_hidden_layers": num_hidden_layers,
            "hidden_dim": hidden_dim
        })

# Run the optimization
DSST_sine_study1.optimize(objective, n_trials=len(DSST_sine_grid_search["num_hidden_layers"]) * len(DSST_sine_grid_search["hidden_dim"]))



""" 2) Algorithm Results """

# Get the grid search results
DSST_sine_grid_results = np.zeros((len(DSST_sine_grid_search["num_hidden_layers"]), len(DSST_sine_grid_search["hidden_dim"])))

for i, num_hidden_layers in enumerate(DSST_sine_grid_search["num_hidden_layers"]):
    for j, hidden_dim in enumerate(DSST_sine_grid_search["hidden_dim"]):
        DSST_sine_trial = DSST_sine_study1.trials[i * len(DSST_sine_grid_search["hidden_dim"]) + j]
        DSST_sine_grid_results[i, j] = 1.0 - DSST_sine_trial.value

# Create a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(DSST_sine_grid_results, cmap="viridis", origin="lower", aspect="auto")
plt.colorbar(label="Objective Value")
plt.xticks(np.arange(len(DSST_sine_grid_results["hidden_dim"])), DSST_sine_grid_results["hidden_dim"])
plt.yticks(np.arange(len(DSST_sine_grid_results["num_hidden_layers"])), DSST_sine_grid_results["num_hidden_layers"])
plt.xlabel("Hidden Dimension")
plt.ylabel("Number of Hidden Layers")
plt.title("Grid Search Results")
plt.grid(False)
plt.savefig("/Sine/DSST/Experiment1/Batch20000_heat.png", dpi=300)


# Select the top 10 trials
DSST_sine_best_trials = DSST_sine_study1.trials_dataframe().sort_values(by='value')[:10]

# Print the information for each of the top trials
with open('/Sine/DSST/Experiment1/Batch20000_top5trials.txt', 'w') as file:
    for i, (_, DSST_sine_trial_row) in enumerate(DSST_sine_best_trials.iterrows()):
        DSST_sine_trial_number = DSST_sine_trial_row['number']
        print(f"Trial {DSST_sine_trial_number}:", file=file)
        print("  Value: ", DSST_sine_trial_row['value'], file=file)
        print("  Params: ", file=file)
        for param_name, param_value in DSST_sine_trial_row.items():
            if param_name.startswith('params_hidden_dim_'):
                dim_number = int(param_name.split('_')[-1])
                print(f"    Hidden Dimension {dim_number}: {param_value}", file=file)
            elif param_name == 'params_num_hidden_layers':
                print(f"    Number of Hidden Layers: {param_value}", file=file)


# Calculate the number of trials
DSST_sine_num_trials = len(DSST_sine_best_trials)

# Calculate the number of rows and columns for subplots
DSST_sine_num_rows = int(np.ceil(np.sqrt(DSST_sine_num_trials)))
DSST_sine_num_cols = int(np.ceil(DSST_sine_num_trials / DSST_sine_num_rows))

# Create subplots
fig, axes = plt.subplots(DSST_sine_num_rows, DSST_sine_num_cols, figsize=(12, 8))

# Plot the training and validation curves for the best trials
for idx, (DSST_sine_trial_number, DSST_sine_trial_row) in enumerate(DSST_sine_best_trials.iterrows()):
    DSST_sine_trial = DSST_sine_study1.trials[DSST_sine_trial_number]
    ax = axes[idx // DSST_sine_num_cols, idx % DSST_sine_num_cols]
    
    # Assuming you have recorded the training and validation metrics during the trials
    DSST_sine_training_curve = DSST_sine_trial.user_attrs['train_losses']
    DSST_sine_validation_curve = DSST_sine_trial.user_attrs['val_losses']
    
    # Plotting the curves
    ax.plot(DSST_sine_training_curve, label='Training')
    ax.plot(DSST_sine_validation_curve, label='Validation')

    # Set the plot title and labels
    ax.set_title(f'Trial {DSST_sine_trial_number}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

# Adjust the spacing between subplots
plt.tight_layout()

# Show or save the plot
plt.savefig("/Sine/DSST/Experiment1/Batch20000_best5.png", dpi=300)
