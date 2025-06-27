# Recurrent Neural Network with PyTorch

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Convert to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_train = X_train.unsqueeze(2)  # shape: (samples, timesteps, 1)

# Part 2 - Building the RNN
class StockPriceRNN(nn.Module):
    def __init__(self):
        super(StockPriceRNN, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.lstm4 = nn.LSTM(input_size=50, hidden_size=50, batch_first=True)
        self.dropout4 = nn.Dropout(0.2)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x, _ = self.lstm4(x)
        x = self.dropout4(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.fc(x)
        return x

# Initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockPriceRNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
batch_size = 32
loss_history = []  # Store loss for each epoch
for epoch in range(num_epochs):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    loss_history.append(epoch_loss)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Part 3 - Making the predictions and visualising the results
# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# Use .to_numpy() to ensure numpy array
total_values = dataset_total.to_numpy().reshape(-1, 1)
inputs = total_values[len(total_values) - len(dataset_test) - 60:]
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = torch.from_numpy(X_test).float().unsqueeze(2).to(device)

model.eval()
with torch.no_grad():
    predicted_stock_price = model(X_test).cpu().numpy()
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction (PyTorch)')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# --- Additional Visualizations ---

# 1. Training Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(loss_history, color='purple')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 2. Zoomed-in Comparison (first 20 days)
plt.figure(figsize=(8, 4))
plt.plot(real_stock_price[:20], color='red', label='Real')
plt.plot(predicted_stock_price[:20], color='blue', label='Predicted')
plt.title('Zoomed-in: First 20 Days')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# 3. Residual Plot
residuals = real_stock_price.flatten() - predicted_stock_price.flatten()
plt.figure(figsize=(8, 4))
plt.plot(residuals, marker='o', linestyle='-', color='green')
plt.title('Residuals (Real - Predicted)')
plt.xlabel('Time')
plt.ylabel('Residual')
plt.grid(True)
plt.show() 