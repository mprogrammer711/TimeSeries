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



def main(file_path, features, target, context_features, n_timesteps, encoding_dim, hidden_dim, lstm_hidden_size, transformer_hidden_size, conv_hidden_size, context_size, output_size, lstm_num_layers, transformer_num_layers, transformer_num_heads, conv_kernel_size, dropout, learning_rate, n_epochs, ae_n_epochs, batch_size, n_splits=5):
    # Load and preprocess data
    df, scaler_target, scaler_context = load_and_preprocess_data(file_path, features, target, context_features)

    # Prepare data
    X, y, context = prepare_data(df, features, target, context_features, n_timesteps)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test, context_train, context_test = train_test_split(X, y, context, test_size=0.2, random_state=42)

    # Train the autoencoder on the entire training set
    train_dataset = TensorDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Define the input dimension, encoding dimension, and hidden dimension
    input_dim = X_train.size(2)

    # Initialize the autoencoder
    autoencoder = Autoencoder(input_dim, encoding_dim, hidden_dim)

    # Define the optimizer and loss function for autoencoder
    ae_optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001)
    ae_criterion = nn.MSELoss()
    ae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(ae_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Train the autoencoder
    train_autoencoder(autoencoder, train_loader, ae_n_epochs, ae_optimizer, ae_criterion, ae_scheduler)

    # Encode the entire training and test data
    encoded_train = encode_data(autoencoder, X_train.view(-1, X_train.size(2)))
    encoded_test = encode_data(autoencoder, X_test.view(-1, X_test.size(2)))

    # Reshape the encoded features back to the original sequence shape
    encoded_train = encoded_train.view(X_train.size(0), X_train.size(1), -1)
    encoded_test = encoded_test.view(X_test.size(0), X_test.size(1), -1)

    # Define K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=False, random_state=42)

    hybrid_train_preds = []
    hybrid_val_preds = []
    true_values = []

    for fold, (train_index, val_index) in enumerate(kf.split(encoded_train)):
        print(f"Fold {fold+1}/{n_splits}")

        X_train_fold, X_val_fold = encoded_train[train_index], encoded_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        context_train_fold, context_val_fold = context_train[train_index], context_train[val_index]

        # Create DataLoader for hybrid model training
        train_dataset = TensorDataset(X_train_fold, y_train_fold, context_train_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(X_val_fold, y_val_fold, context_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_size = X_train_fold.size(2)  # Encoded feature size
        model = HybridForecastingModel(input_size, conv_hidden_size, lstm_hidden_size, transformer_hidden_size, context_size, output_size, lstm_num_layers, transformer_num_layers, transformer_num_heads, conv_kernel_size, dropout)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Train the hybrid model with early stopping
        train_hybrid(model, train_loader, val_loader, n_epochs, optimizer, criterion, scheduler)

        # Get hybrid model predictions for XGBoost training
        model.eval()
        with torch.no_grad():
            fold_train_preds = []
            for batch in train_loader:
                past_data, _, context_data = batch
                hybrid_output = model(past_data, context_data)
                fold_train_preds.append(hybrid_output.cpu().numpy())
            fold_train_preds = np.concatenate(fold_train_preds, axis=0)
            hybrid_train_preds.append(fold_train_preds)

            fold_val_preds = []
            for batch in val_loader:
                past_data, _, context_data = batch
                hybrid_output = model(past_data, context_data)
                fold_val_preds.append(hybrid_output.cpu().numpy())
            fold_val_preds = np.concatenate(fold_val_preds, axis=0)
            hybrid_val_preds.append(fold_val_preds)

        # Store true values for evaluation
        true_values.append(y_val_fold.numpy())

    # Concatenate all predictions and true values
    hybrid_train_preds = np.concatenate(hybrid_train_preds, axis=0)
    hybrid_val_preds = np.concatenate(hybrid_val_preds, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    # Train XGBoost model on the entire training set
    dtrain = xgb.DMatrix(hybrid_train_preds, label=y_train.numpy())
    dval = xgb.DMatrix(hybrid_val_preds, label=true_values)
    evals = [(dtrain, 'train'), (dval, 'eval')]

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.01,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    xgb_model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=10, verbose_eval=True)

    # Evaluate the XGBoost model on the test set
    test_dataset = TensorDataset(encoded_test, y_test, context_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    with torch.no_grad():
        hybrid_test_preds = []
        for batch in test_loader:
            past_data, _, context_data = batch
            hybrid_output = model(past_data, context_data)
            hybrid_test_preds.append(hybrid_output.cpu().numpy())
        hybrid_test_preds = np.concatenate(hybrid_test_preds, axis=0)

    dtest = xgb.DMatrix(hybrid_test_preds)
    predictions = xgb_model.predict(dtest)

    # Inverse transform the predictions to the original scale
    predictions_original_scale = scaler_target.inverse_transform(predictions.reshape(-1, 1))
    true_values_original_scale = scaler_target.inverse_transform(y_test.numpy().reshape(-1, 1))

    # Print or return the results as needed
    print(f"Predictions: {predictions_original_scale}")
    print(f"True Values: {true_values_original_scale}")

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
import numpy as np
import matplotlib.pyplot as plt

# Load the data
file_path = 'your_data.csv'
df = pd.read_csv(file_path)

# Define the target feature
target_feature = 'target'

# Define parameters for momentum and moving average
momentum_window = 5
moving_average_window = 10
n_steps = 10

# Calculate momentum
df['momentum'] = df[target_feature].diff(momentum_window)

# Calculate moving average
df['moving_average'] = df[target_feature].rolling(window=moving_average_window).mean()

# Drop NaN values created by diff and rolling
df.dropna(inplace=True)

# Forecast the next n steps using the real previous data
def forecast_momentum(df, target_feature, n_steps, momentum_window):
    forecasts = []
    for i in range(n_steps):
        # Calculate momentum for the current step
        momentum = df[target_feature].iloc[-momentum_window:].diff().sum()
        
        # Forecast the next value
        forecast = df[target_feature].iloc[-1] + momentum
        
        # Append the forecast to the list
        forecasts.append(forecast)
        
        # Append the forecast to the dataframe to use it for the next step
        df = df.append({target_feature: forecast}, ignore_index=True)
    
    return forecasts

def forecast_moving_average(df, target_feature, n_steps, moving_average_window):
    forecasts = []
    for i in range(n_steps):
        # Calculate moving average for the current step
        moving_average = df[target_feature].iloc[-moving_average_window:].mean()
        
        # Forecast the next value
        forecast = moving_average
        
        # Append the forecast to the list
        forecasts.append(forecast)
        
        # Append the forecast to the dataframe to use it for the next step
        df = df.append({target_feature: forecast}, ignore_index=True)
    
    return forecasts

# Perform the forecasting
momentum_forecasts = forecast_momentum(df.copy(), target_feature, n_steps, momentum_window)
moving_average_forecasts = forecast_moving_average(df.copy(), target_feature, n_steps, moving_average_window)

# Print the forecasts
print(f'Forecasts for the next {n_steps} steps using momentum strategy:')
print(momentum_forecasts)
print(f'Forecasts for the next {n_steps} steps using moving average strategy:')
print(moving_average_forecasts)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(df[target_feature], label='Real Data')
plt.plot(range(len(df), len(df) + n_steps), momentum_forecasts, label='Momentum Forecasts', color='red')
plt.plot(range(len(df), len(df) + n_steps), moving_average_forecasts, label='Moving Average Forecasts', color='green')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Real Data and Forecasts')
plt.show()


import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the folder containing the forecast CSV files
folder_path = 'forecast'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Dictionary to store aggregated data for each instrument
instrument_data = {}

# Iterate through each CSV file
for csv_file in csv_files:
    # Extract the date from the file name
    date = csv_file.split('_')[1].split('.')[0]
    
    # Read the CSV file into a DataFrame
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    
    # Print the columns of the DataFrame
    print(f"Columns of the DataFrame for {csv_file}:")
    print(df.columns)
    
    # Add the date column to the DataFrame
    df['date'] = date
    
    # Calculate the differences for each forecast column and round to 2 decimal places
    df['diff_forecast_lstm'] = (df['forecast_lstm'] - df['close yesterday']).round(2)
    df['diff_forecast_momentum_strategy'] = (df['forecast_momentum_strategy'] - df['close yesterday']).round(2)
    df['diff_forecast_moving_average'] = (df['forecast_moving_average'] - df['close yesterday']).round(2)
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        instrument = row['real instrument name']
        
        # If the instrument is not already in the dictionary, initialize it
        if instrument not in instrument_data:
            instrument_data[instrument] = []
        
        # Append the row data to the instrument's list
        instrument_data[instrument].append(row)

# Save the aggregated data for each instrument to separate files
output_folder = 'aggregated_forecasts'
os.makedirs(output_folder, exist_ok=True)

plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)

for instrument, data in instrument_data.items():
    # Convert the list of rows to a DataFrame
    instrument_df = pd.DataFrame(data)
    
    # Define the output file path
    output_file = os.path.join(output_folder, f'{instrument}_forecasts.csv')
    
    # Save the DataFrame to a CSV file
    instrument_df.to_csv(output_file, index=False)
    
    # Plot the differences
    plt.figure(figsize=(10, 6))
    plt.plot(instrument_df['date'], instrument_df['diff_forecast_lstm'], label='LSTM Difference', marker='o')
    plt.plot(instrument_df['date'], instrument_df['diff_forecast_momentum_strategy'], label='Momentum Strategy Difference', marker='o')
    plt.plot(instrument_df['date'], instrument_df['diff_forecast_moving_average'], label='Moving Average Difference', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.title(f'Differences for {instrument}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(plot_folder, f'{instrument}_differences.png')
    plt.savefig(plot_file)
    plt.close()

print("Aggregated forecasts and plots saved successfully.")







import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Define the folder containing the forecast CSV files
folder_path = 'forecast'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Dictionary to store aggregated data for each instrument
instrument_data = {}

# Iterate through each CSV file
for csv_file in csv_files:
    # Extract the date from the file name
    date = csv_file.split('_')[1].split('.')[0]
    
    # Read the CSV file into a DataFrame
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    
    # Print the columns of the DataFrame
    print(f"Columns of the DataFrame for {csv_file}:")
    print(df.columns)
    
    # Add the date column to the DataFrame
    df['date'] = date
    
    # Calculate the differences for each forecast column and round to 2 decimal places
    df['diff_forecast_lstm'] = (df['forecast_lstm'] - df['close yesterday']).round(2)
    df['diff_forecast_momentum_strategy'] = (df['forecast_momentum_strategy'] - df['close yesterday']).round(2)
    df['diff_forecast_moving_average'] = (df['forecast_moving_average'] - df['close yesterday']).round(2)
    
    # Calculate daily returns for the instrument
    df['daily_return'] = df['close today'] / df['close yesterday'] - 1
    
    # Calculate the hit ratio for each forecast method
    df['hit_lstm'] = np.sign(df['forecast_lstm'] - df['close yesterday']) == np.sign(df['close today'] - df['close yesterday'])
    df['hit_momentum_strategy'] = np.sign(df['forecast_momentum_strategy'] - df['close yesterday']) == np.sign(df['close today'] - df['close yesterday'])
    df['hit_moving_average'] = np.sign(df['forecast_moving_average'] - df['close yesterday']) == np.sign(df['close today'] - df['close yesterday'])
    
    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        instrument = row['real instrument name']
        
        # If the instrument is not already in the dictionary, initialize it
        if instrument not in instrument_data:
            instrument_data[instrument] = []
        
        # Append the row data to the instrument's list
        instrument_data[instrument].append(row)

# Save the aggregated data for each instrument to separate files
output_folder = 'aggregated_forecasts'
os.makedirs(output_folder, exist_ok=True)

plot_folder = 'plots'
os.makedirs(plot_folder, exist_ok=True)

# Define your portfolio with instrument weights
portfolio_weights = {
    'Instrument1': 0.4,
    'Instrument2': 0.3,
    'Instrument3': 0.3
}

# List to store portfolio returns
portfolio_returns = []

# Dictionary to store hit ratios
hit_ratios = {
    'lstm': [],
    'momentum_strategy': [],
    'moving_average': []
}

# Dictionary to store PnL for each strategy
pnl = {
    'lstm': [],
    'momentum_strategy': [],
    'moving_average': []
}

for instrument, data in instrument_data.items():
    # Convert the list of rows to a DataFrame
    instrument_df = pd.DataFrame(data)
    
    # Define the output file path
    output_file = os.path.join(output_folder, f'{instrument}_forecasts.csv')
    
    # Save the DataFrame to a CSV file
    instrument_df.to_csv(output_file, index=False)
    
    # Plot the differences
    plt.figure(figsize=(10, 6))
    plt.plot(instrument_df['date'], instrument_df['diff_forecast_lstm'], label='LSTM Difference', marker='o')
    plt.plot(instrument_df['date'], instrument_df['diff_forecast_momentum_strategy'], label='Momentum Strategy Difference', marker='o')
    plt.plot(instrument_df['date'], instrument_df['diff_forecast_moving_average'], label='Moving Average Difference', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Difference')
    plt.title(f'Differences for {instrument}')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(plot_folder, f'{instrument}_differences.png')
    plt.savefig(plot_file)
    plt.close()
    
    # Add the weighted daily returns to the portfolio returns
    if instrument in portfolio_weights:
        weighted_returns = instrument_df['daily_return'] * portfolio_weights[instrument]
        portfolio_returns.append(weighted_returns)
    
    # Calculate hit ratios for each forecast method
    hit_ratios['lstm'].append(instrument_df['hit_lstm'].mean())
    hit_ratios['momentum_strategy'].append(instrument_df['hit_momentum_strategy'].mean())
    hit_ratios['moving_average'].append(instrument_df['hit_moving_average'].mean())
    
    # Calculate PnL for each strategy
    instrument_df['pnl_lstm'] = (instrument_df['forecast_lstm'] - instrument_df['close yesterday']) * instrument_df['daily_return']
    instrument_df['pnl_momentum_strategy'] = (instrument_df['forecast_momentum_strategy'] - instrument_df['close yesterday']) * instrument_df['daily_return']
    instrument_df['pnl_moving_average'] = (instrument_df['forecast_moving_average'] - instrument_df['close yesterday']) * instrument_df['daily_return']
    
    pnl['lstm'].append(instrument_df['pnl_lstm'])
    pnl['momentum_strategy'].append(instrument_df['pnl_momentum_strategy'])
    pnl['moving_average'].append(instrument_df['pnl_moving_average'])

# Combine the portfolio returns into a single DataFrame
portfolio_returns_df = pd.concat(portfolio_returns, axis=1)
portfolio_returns_df['portfolio_return'] = portfolio_returns_df.sum(axis=1)

# Calculate the cumulative returns of the portfolio
portfolio_returns_df['cumulative_return'] = (1 + portfolio_returns_df['portfolio_return']).cumprod()

# Calculate the running maximum of the cumulative returns
portfolio_returns_df['running_max'] = portfolio_returns_df['cumulative_return'].cummax()

# Calculate the drawdown
portfolio_returns_df['drawdown'] = portfolio_returns_df['running_max'] - portfolio_returns_df['cumulative_return']
portfolio_returns_df['drawdown_pct'] = portfolio_returns_df['drawdown'] / portfolio_returns_df['running_max']

# Calculate the average portfolio return and standard deviation
average_portfolio_return = portfolio_returns_df['portfolio_return'].mean()
portfolio_return_std = portfolio_returns_df['portfolio_return'].std()

# Define the risk-free rate (e.g., 0.01 for 1%)
risk_free_rate = 0.01

# Calculate the Sharpe ratio
sharpe_ratio = (average_portfolio_return - risk_free_rate) / portfolio_return_std

# Calculate the overall hit ratios
overall_hit_ratios = {key: np.mean(values) for key, values in hit_ratios.items()}

# Combine the PnL for each strategy into a single DataFrame
pnl_df = pd.DataFrame({
    'lstm': pd.concat(pnl['lstm']).reset_index(drop=True),
    'momentum_strategy': pd.concat(pnl['momentum_strategy']).reset_index(drop=True),
    'moving_average': pd.concat(pnl['moving_average']).reset_index(drop=True)
})

# Calculate the cumulative PnL for each strategy
pnl_df['cumulative_pnl_lstm'] = pnl_df['lstm'].cumsum()
pnl_df['cumulative_pnl_momentum_strategy'] = pnl_df['momentum_strategy'].cumsum()
pnl_df['cumulative_pnl_moving_average'] = pnl_df['moving_average'].cumsum()

# Calculate the total PnL for each strategy
total_pnl = {
    'lstm': pnl_df['lstm'].sum(),
    'momentum_strategy': pnl_df['momentum_strategy'].sum(),
    'moving_average': pnl_df['moving_average'].sum()
}

# Calculate Alpha and Beta
benchmark_returns = portfolio_returns_df['portfolio_return']  # Assuming the portfolio return is the benchmark
X = sm.add_constant(benchmark_returns)

alphas = {}
betas = {}
for strategy in ['lstm', 'momentum_strategy', 'moving_average']:
    y = pnl_df[strategy]
    model = sm.OLS(y, X).fit()
    alphas[strategy] = model.params['const']
    betas[strategy] = model.params['portfolio_return']

# Calculate Information Ratio
information_ratios = {}
for strategy in ['lstm', 'momentum_strategy', 'moving_average']:
    excess_return = pnl_df[strategy] - benchmark_returns
    tracking_error = excess_return.std()
    information_ratios[strategy] = excess_return.mean() / tracking_error

# Calculate Volatility
volatility = {}
for strategy in ['lstm', 'momentum_strategy', 'moving_average']:
    volatility[strategy] = pnl_df[strategy].std()

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Overall Hit Ratios: {overall_hit_ratios}")
print(f"Total PnL: {total_pnl}")
print(f"Alphas: {alphas}")
print(f"Betas: {betas}")
print(f"Information Ratios: {information_ratios}")
print(f"Volatility: {volatility}")

# Plot the drawdown
plt.figure(figsize=(10, 6))
plt.plot(portfolio_returns_df.index, portfolio_returns_df['drawdown_pct'], label='Drawdown', color='red')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.title('Portfolio Drawdown')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save the drawdown plot
drawdown_plot_file = os.path.join(plot_folder, 'portfolio_drawdown.png')
plt.savefig(drawdown_plot_file)
plt.close()

# Plot the cumulative PnL for each strategy
plt.figure(figsize=(10, 6))
plt.plot(pnl_df.index, pnl_df['cumulative_pnl_lstm'], label='Cumulative PnL LSTM', marker='o')
plt.plot(pnl_df.index, pnl_df['cumulative_pnl_momentum_strategy'], label='Cumulative PnL Momentum Strategy', marker='o')
plt.plot(pnl_df.index, pnl_df['cumulative_pnl_moving_average'], label='Cumulative PnL Moving Average', marker='o')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL')
plt.title('Cumulative PnL for Each Strategy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Save the cumulative PnL plot
cumulative_pnl_plot_file = os.path.join(plot_folder, 'cumulative_pnl.png')
plt.savefig(cumulative_pnl_plot_file)
plt.close()

print("Aggregated forecasts, plots, drawdown, and PnL saved successfully.")
