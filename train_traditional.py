import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
import matplotlib.pyplot as plt
import pickle
import torch
import numpy as np

import torch.nn.functional as F
import random

def load_from_file(a_path: str):
    # open a file, where you stored the pickled data
    pkl_file = open(a_path, 'rb')

    # dump information to that file
    gei_data = pickle.load(pkl_file)
    pkl_file.close()
    return gei_data

def save_to_file(dataset, output_file):
    with open(output_file, 'wb') as fp:
        pickle.dump(dataset, fp)
    pass

import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/kaggle/input/vpb-data-credit-score-etl/tradition.csv')

df = df.iloc[:, 1:]

train_df = df.iloc[0:int(0.9 * len(df)) ,:]
valid_df = df.iloc[int(0.9 * len(df)): ,:]

X = train_df.iloc[:, 1:]
y = train_df['CREDIT_SCORE']

X_valid = valid_df.iloc[:, 1:]
y_valid = valid_df['CREDIT_SCORE']

X = X.to_numpy()
y = y.to_numpy()

X_valid = X_valid.to_numpy()
y_valid = y_valid.to_numpy()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import numpy as np
score_scaler = MinMaxScaler()

# A FICO score is a three-digit number between 300 and 850
#that measures the creditworthiness

manual_min = np.array([300])
manual_max = np.array([850])

combined_data = np.vstack((y.reshape(-1, 1), manual_min, manual_max))

# We will use the MinMax scaler to transform the target variable Y to the range [0,1].
score_scaler.fit(combined_data)


y_scaled = score_scaler.transform(y.reshape(-1, 1)).flatten()
y_valid_scaled = score_scaler.transform(y_valid.reshape(-1, 1)).flatten()

print(f"Per-feature minimum seen in the data (data_min_): {score_scaler.data_min_}")
print(f"Per-feature maximum seen in the data (data_max_): {score_scaler.data_max_}")
print(f"Per-feature range (data_max_ - data_min_) seen in the data (data_range_): {score_scaler.data_range_}")
print(f"Per-feature relative scaling of the data (scale_): {score_scaler.scale_}")
print(f"Per-feature adjustment for minimum (min_): {score_scaler.min_}")
print(f"Number of features seen during fit (n_features_in_): {score_scaler.n_features_in_}")

class CreditDataset(Dataset):
    def __init__(self, features, labels):
        # Convert NumPy arrays to PyTorch Tensors
        self.features = torch.tensor(features, dtype=torch.float32)
        # unsqueeze for (N, 1) shape
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


INPUT_SIZE = len(X[0])    # number of features
DROPOUT=0.4 #0.4 #0.3 #0.4 #0.5
INIT_DROPOUT=0.2 #0.05

OUTPUT_SIZE = 1    # Predicting a single continuous credit score
LEARNING_RATE = 0.0001 #0.001
BATCH_SIZE = 18 #int(len(X_train) / 20)
VALID_BATCH_SIZE = len(X_valid)
NUM_EPOCHS = 4000 #10000 #200 #100 #200 #5000# 10000

class CreditScoreRegressor(nn.Module):
    def __init__(self, input_size=INPUT_SIZE,
                 output_size=OUTPUT_SIZE, dropout=DROPOUT):
        super(CreditScoreRegressor, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(512,  256),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(dropout),

            nn.Linear(64,output_size)
        )

    def forward(self, x):

        return self.fc1(x)

# Instantiate the model
model = CreditScoreRegressor()
train_dataset = CreditDataset(X, y_scaled)
valid_dataset = CreditDataset(X_valid, y_valid_scaled)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

import numpy as np
train_loss = np.inf
valid_loss = np.inf

train_loss_list = []
valid_loss_list = []

init_dropout = nn.Dropout(INIT_DROPOUT)

# --- Training Loop ---
print("Starting Model Training for Regression...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for i, (features, labels) in enumerate(train_loader):

        feature = init_dropout(features)
        # Forward pass
        outputs = model(feature)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    a_loss = total_loss / len(train_loader)
    train_loss_list.append(a_loss)

    if a_loss <= train_loss:
        train_loss = a_loss
        torch.save(model.state_dict(), f'best_model_train.pt')
        pass

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, (features, labels) in enumerate(valid_loader):

            # Forward pass
            outputs = model(features)
            vloss = criterion(outputs, labels)

            total_loss += vloss.item()

    a_loss = total_loss / len(valid_loader)
    valid_loss_list.append(a_loss)

    if a_loss <= valid_loss:
        valid_loss = a_loss
        torch.save(model.state_dict(), f'best_model_valid.pt')
        pass

    # Print training stats every 10 epochs
    if (epoch + 1) % 500 == 0:
        # The loss here is the MSE, which is the squared error
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], MSE Train Loss: {loss.item():.6f} , MSE valid Loss: {vloss.item():.6f}')
        print('train loss', train_loss)
        print('valid loss', valid_loss)



print("Training Complete.")
