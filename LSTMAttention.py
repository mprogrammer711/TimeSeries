import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.attn.weight)
        nn.init.constant_(self.attn.bias, 0)
        nn.init.uniform_(self.v, -0.1, 0.1)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        attention_weights = torch.bmm(v, energy).squeeze(1)
        return torch.softmax(attention_weights, dim=1)

# Define LSTM Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# Define LSTM Decoder with Attention
class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, context_size, output_size, num_layers, dropout):
        super(LSTMDecoderWithAttention, self).__init__()
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2 + context_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, hidden, cell, context_data):
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        context_combined = torch.cat((context, context_data), dim=1).unsqueeze(1)  # Add sequence dimension
        outputs, (hidden, cell) = self.lstm(context_combined, (hidden, cell))
        prediction = self.fc(outputs.squeeze(1))
        return prediction, hidden, cell, attn_weights

# Define the Hybrid Model
class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, context_size, output_size, num_layers, dropout):
        super(HybridModel, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoderWithAttention(hidden_size, context_size, output_size, num_layers, dropout)

    def forward(self, past_data, context_data):
        encoder_outputs, hidden, cell = self.encoder(past_data)
        prediction, _, _, attn_weights = self.decoder(encoder_outputs, hidden, cell, context_data)
        return prediction, attn_weights

def load_and_preprocess_data(file_path, features, target, context_features):
    df = pd.read_csv(file_path)

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill').fillna(method='bfill')

    # Normalize data
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    scaler_context = MinMaxScaler()

    # Normalize feature columns
    df[features] = scaler_features.fit_transform(df[features])

    # Normalize target columns
    df[target] = scaler_target.fit_transform(df[target])

    # Normalize context columns
    df[context_features] = scaler_context.fit_transform(df[context_features])

    return df, scaler_target, scaler_context

def create_sequences(data, target_data, context_data, n_timesteps):
    X, y, context_features = [], [], []
    for i in range(len(data) - n_timesteps):
        X.append(data[i:i + n_timesteps].values)
        y.append(target_data.iloc[i + n_timesteps].values)
        context_features.append(context_data.iloc[i + n_timesteps].values)
    return np.array(X), np.array(y), np.array(context_features)

def prepare_data(df, features, target, context_features, n_timesteps, test_size=0.2):
    X, y, context = create_sequences(df[features], df[target], df[context_features], n_timesteps)
    # Set shuffle=False in train_test_split to preserve temporal order
    X_train, X_test, y_train, y_test, context_train, context_test = train_test_split(
        X, y, context, test_size=test_size, shuffle=False)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    context_train = torch.tensor(context_train, dtype=torch.float32)
    context_test = torch.tensor(context_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test, context_train, context_test

def train_model(model, train_loader, val_loader, n_epochs, optimizer, criterion, scheduler, patience):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            past_data, y_batch, context_data = batch
            optimizer.zero_grad()
            outputs, _ = model(past_data, context_data)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                past_data, y_batch, context_data = batch
                outputs, _ = model(past_data, context_data)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def evaluate_model(model, test_loader, scaler_target):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    predictions, true_values, attention_weights = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            past_data, y_batch, context_data = batch
            outputs, attn_weights = model(past_data, context_data)
            predictions.append(outputs.cpu().numpy())
            true_values.append(y_batch.cpu().numpy())
            attention_weights.append(attn_weights.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    attention_weights = np.concatenate(attention_weights, axis=0)

    predictions_original_scale = scaler_target.inverse_transform(predictions)
    true_values_original_scale = scaler_target.inverse_transform(true_values)

    mse = mean_squared_error(true_values_original_scale, predictions_original_scale)
    r2 = r2_score(true_values_original_scale, predictions_original_scale)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    return predictions_original_scale, true_values_original_scale, attention_weights

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('True Values vs Predictions')
    plt.show()

def plot_attention_weights(attention_weights, sample_idx=0):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights[sample_idx], cmap='viridis')
    plt.xlabel('Encoder Time Steps')
    plt.ylabel('Attention Weights')
    plt.title('Attention Weights for Sample Index {}'.format(sample_idx))
    plt.show()

def main(file_path, features, target, context_features, n_timesteps, n_epochs, batch_size, learning_rate, patience):
    # Load and preprocess
