import pandas as pd

df = df.drop(df.columns[[0, 1]], axis=1)


# Read the CSV file
df = pd.read_csv('time_series.csv', parse_dates=['Time'], index_col='Time')

# Define a function to replace outliers with the previous valid value
def replace_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = pd.NA
    df[column].fillna(method='pad', inplace=True)
    return df

# Replace outliers in each column
for column in df.columns:
    df = replace_outliers(df, column)

import pandas as pd

# Read the CSV file
df = pd.read_csv('time_series.csv', parse_dates=['Time'], index_col='Time')

# Check for duplicate rows
duplicates = df.duplicated(keep=False)

# Print the rows where the data stays the same for a period of time
print(df[duplicates])

import pandas as pd

# Read the CSV file
df = pd.read_csv('time_series.csv', parse_dates=['Time'], index_col='Time')

# For each column, print the unique values and their counts
for column in df.columns:
    print(f"{column}:\n{df[column].value_counts()}\n")

import pandas as pd

# Read the CSV file
df = pd.read_csv('time_series.csv', parse_dates=['Time'], index_col='Time')

# Resample to daily frequency
df_daily = df.resample('D').last()

# Create a new DataFrame to store whether each feature has changed each day
df_changes = df_daily.diff().notna()

# Print the new DataFrame
print(df_changes)


import pandas as pd
from datetime import datetime

# Path to the Excel file
file_path = 'ile.xlsx'

# Read the Excel file
excel_file = pd.ExcelFile(file_path)

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each sheet in the Excel file
for sheet_name in excel_file.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Identify the location of 'last Price'
    last_price_location = df.isin(['last Price']).stack().idxmax()
    last_price_row_idx = last_price_location[0]
    last_price_col_idx = last_price_location[1]
    
    # Extract stock names (row just above 'last Price')
    stock_names = df.iloc[last_price_row_idx - 1, last_price_col_idx + 1:].values
    
    # Extract dates (column just before 'last Price' row)
    dates = df.iloc[last_price_row_idx + 1:, last_price_col_idx].values
    
    # Extract values (rows starting from 'last Price' row)
    values = df.iloc[last_price_row_idx + 1:, last_price_col_idx + 1:].values
    
    # Create a new DataFrame with the extracted data
    extracted_df = pd.DataFrame(values, columns=stock_names, index=dates)
    
    # Append the extracted DataFrame to the list
    dfs.append(extracted_df)

# Concatenate all DataFrames into a single DataFrame
final_df = pd.concat(dfs, axis=1)

# Reset the index to have dates as a column
final_df.reset_index(inplace=True)
final_df.rename(columns={'index': 'Date'}, inplace=True)

# Display the final DataFrame
print(final_df)

# Save the final DataFrame to a CSV file with today's date
today = datetime.today()
date_str = today.strftime('%Y-%m-%d')
filename = f"data_{date_str}.csv"
final_df.to_csv(filename, index=False)

print(f"DataFrame saved to {filename}")



import pandas as pd

# Sample data
data = {
    'cut_id': ['20230101', '20230201', '20230301', '20230401123045']
}
df = pd.DataFrame(data)

# Custom function to parse dates
def parse_date(date_str):
    if len(date_str) == 8:
        return pd.to_datetime(date_str, format='%Y%m%d')
    elif len(date_str) == 14:
        return pd.to_datetime(date_str, format='%Y%m%d%H%M%S').date()
    else:
        raise ValueError(f"Unexpected date format: {date_str}")

# Apply the custom function to the 'cut_id' column
df['cut_id'] = df['cut_id'].apply(parse_date)

print(df)

import xgboost as xgb
from xgboost.callback import TrainingCallback

class AdaptiveLearningRate(TrainingCallback):
    def __init__(self, initial_lr, decay_factor, decay_step):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.decay_step = decay_step

    def after_iteration(self, model, epoch, evals_log):
        if epoch % self.decay_step == 0 and epoch != 0:
            new_lr = self.initial_lr * (self.decay_factor ** (epoch // self.decay_step))
            model.set_param('learning_rate', new_lr)
            print(f"Updated learning rate to {new_lr:.6f} at epoch {epoch}")
        return False

# Define the XGBoost model with adaptive learning rate
initial_lr = 0.1
decay_factor = 0.9
decay_step = 10

xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=initial_lr,
    tree_method='hist'
)

# Fit the model with the adaptive learning rate callback
xgb_model.fit(
    X_train,
    y_train,
    callbacks=[AdaptiveLearningRate(initial_lr, decay_factor, decay_step)]
)


import pandas as pd

# Load the main CSV file
main_df = pd.read_csv('main_data.csv', parse_dates=['date'])

# Load the historical CSV file
history_df = pd.read_csv('history_data.csv', parse_dates=['date'])

# Merge the DataFrames on the date column
combined_df = pd.merge(main_df, history_df, on='date', how='left', suffixes=('', '_history'))

# Fill missing values in the main DataFrame using the historical data
for feature in ['feature1', 'feature2']:
    combined_df[feature].fillna(combined_df[f'{feature}_history'], inplace=True)

# Drop the historical columns
combined_df.drop(columns=[f'{feature}_history' for feature in ['feature1', 'feature2']], inplace=True)






import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

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

# Define Encoder
class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell

# Define Decoder with Attention
class LSTMDecoderWithAttention(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(LSTMDecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, hidden, cell, encoder_outputs):
        attention_weights = self.attention(hidden[-1], encoder_outputs)
        attention_weights = attention_weights.unsqueeze(1)
        context = torch.bmm(attention_weights, encoder_outputs)
        decoder_input = torch.cat([context, hidden[-1].unsqueeze(1)], dim=2)
        output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
        prediction = self.fc(output).squeeze(1)
        return prediction, hidden, cell, attention_weights.squeeze(1)

# Define full Encoder-Decoder Model
class LSTMEncoderDecoderWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(LSTMEncoderDecoderWithAttention, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = LSTMDecoderWithAttention(hidden_size, output_size, num_layers, dropout)

    def forward(self, encoder_input):
        encoder_outputs, hidden, cell = self.encoder(encoder_input)
        prediction, _, _, attention_weights = self.decoder(hidden, cell, encoder_outputs)
        return prediction, attention_weights

def load_and_preprocess_data(file_path, features, target):
    df = pd.read_csv(file_path)

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        df = df.fillna(method='ffill').fillna(method='bfill')

    # Normalize data
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # Normalize feature columns
    df[features] = scaler_features.fit_transform(df[features])

    # Normalize target columns
    df[target] = scaler_target.fit_transform(df[target])

    return df, scaler_target

def create_sequences(data, target_data, n_timesteps):
    X, y = [], []
    for i in range(len(data) - n_timesteps):
        X.append(data[i:i + n_timesteps].values)
        y.append(target_data.iloc[i + n_timesteps].values)
    return np.array(X), np.array(y)

def prepare_data(df, features, target, n_timesteps, test_size=0.2):
    X, y = create_sequences(df[features], df[target], n_timesteps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, X_test, y_train, y_test

def train_lstm(model, train_loader, n_epochs, optimizer, criterion, scheduler=None):
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X_batch, y_batch = batch
            output, _ = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        if scheduler:
            scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def evaluate_lstm(model, test_loader, scaler_target):
    model.eval()
    forecast_list = []
    attention_weights_list = []
    with torch.no_grad():
        for batch in test_loader:
            X_batch, _ = batch
            forecast, attention_weights = model(X_batch)
            forecast_list.append(forecast.cpu().numpy())
            attention_weights_list.append(attention_weights.cpu().numpy())

    # Convert lists to numpy arrays
    forecast_array = np.concatenate(forecast_list, axis=0)
    attention_weights_array = np.concatenate(attention_weights_list, axis=0)

    # Inverse-transform the predictions to the original scale
    forecast_original_scale = scaler_target.inverse_transform(forecast_array)
    return forecast_original_scale, attention_weights_array

def visualize_attention_weights(attention_weights_array, sample_index=0):
    attention_weights_sample = attention_weights_array[sample_index]
    plt.figure(figsize=(10, 6))
    sns.heatmap(attention_weights_sample.reshape(1, -1), cmap='viridis', annot=True)
    plt.title('Attention Weights for Sample Index {}'.format(sample_index))
    plt.xlabel('Input Sequence Index')
    plt.ylabel('Attention Weight')
    plt.show()

def main(file_path, features, target, n_timesteps, hidden_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout, lstm_learning_rate, lstm_n_epochs, batch_size):
    # Load and preprocess data
    df, scaler_target = load_and_preprocess_data(file_path, features, target)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df, features, target, n_timesteps)

    # Create DataLoader for LSTM training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = X_train.size(2)  # Feature size
    output_size = y_train.size(1)  # Number of targets
    model = LSTMEncoderDecoderWithAttention(input_size, lstm_hidden_size, output_size, lstm_num_layers, lstm_dropout)
    optimizer = optim.Adam(model.parameters(), lr=lstm_learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the LSTM model
    train_lstm(model, train_loader, lstm_n_epochs, optimizer, criterion, scheduler)

    # Evaluate the LSTM model
    forecast_original_scale, attention_weights_array = evaluate_lstm(model, test_loader, scaler_target)

    # Compare predictions to the actual values
    y_test_original_scale = scaler_target.inverse_transform(y_test.cpu().numpy())
    print("Predictions on original scale:", forecast_original_scale)
    print("True values on original scale:", y_test_original_scale)

    # Visualize attention weights for a specific sample
    visualize_attention_weights(attention_weights_array)

# Define parameters
file_path = 'time_series_data.csv'
features = ['rate_level_1', 'rate_level_2', 'days_to_end_of_month', 'days_to_ECB_meeting', 'days_to_Fed_meeting', 'ois_sofr_rate']
target = ['rate_level_1', 'rate_level_2']
n_timesteps = 12
hidden_dim = 128
lstm_hidden_size = 128
lstm_num_layers = 2
lstm_dropout = 0.3
lstm_learning_rate = 0.001
lstm_n_epochs = 50
batch_size = 64

# Run the main function
main(file_path, features, target, n_timesteps, hidden_dim, lstm_hidden_size, lstm_num_layers, lstm_dropout, lstm_learning_rate, lstm_n_epochs, batch_size)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)

print(combined_df.head())
