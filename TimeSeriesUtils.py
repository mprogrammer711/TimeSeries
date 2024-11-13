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



hybrid_predictions = []
    true_values = []
    hybrid_train_preds = []
    hybrid_val_preds = []

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

        # Initialize hybrid model
        input_size = X_train_fold.size(2)  # Encoded feature size
        model = HybridForecastingModel(input_size, conv_hidden_size, lstm_hidden_size, transformer_hidden_size, context_size, output_size, lstm_num_layers, transformer_num_layers, transformer_num_heads, conv_kernel_size, dropout)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # L2 regularization
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

        # Train the hybrid model with early stopping
        model = train_hybrid(model, train_loader, val_loader, n_epochs, optimizer, criterion, scheduler)

        # Get hybrid model predictions for training and validation sets
        model.eval()
        with torch.no_grad():
            # Get predictions for training set
            hybrid_train_preds_fold = []
            for batch in train_loader:
                past_data, _, context_data = batch
                hybrid_output = model(past_data, context_data)
                hybrid_train_preds_fold.append(hybrid_output.cpu().numpy())
            hybrid_train_preds_fold = np.concatenate(hybrid_train_preds_fold, axis=0)

            # Get predictions for validation set
            hybrid_val_preds_fold = []
            for batch in val_loader:
                past_data, _, context_data = batch
                hybrid_output = model(past_data, context_data)
                hybrid_val_preds_fold.append(hybrid_output.cpu().numpy())
            hybrid_val_preds_fold = np.concatenate(hybrid_val_preds_fold, axis=0)

        hybrid_train_preds.append(hybrid_train_preds_fold)
        hybrid_val_preds.append(hybrid_val_preds_fold)
        true_values.append(y_val_fold.numpy())

    # Concatenate all hybrid model predictions and true values
    hybrid_train_preds = np.concatenate(hybrid_train_preds, axis=0)
    hybrid_val_preds = np.concatenate(hybrid_val_preds, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    # Train XGBoost model using hybrid model predictions as features
    print("Training XGBoost on hybrid model predictions...")
    xgb_model = train_xgboost_with_callback(hybrid_train_preds, y_train.numpy(), hybrid_val_preds, true_values)

    # Evaluate the XGBoost model on the test set
    dtest = xgb.DMatrix(hybrid_val_preds)  # Using hybrid model predictions as input
    xgb_predictions = xgb_model.predict(dtest)

    # Inverse transform the predictions to the original scale
    xgb_predictions_original_scale = scaler_target.inverse_transform(xgb_predictions.reshape(-1, 1))
    true_values_original_scale = scaler_target.inverse_transform(true_values.reshape(-1, 1))


# Define Convolutional Layers with Batch Normalization using nn.Sequential
class ConvLayers(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dropout):
        super(ConvLayers, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, input_size, seq_len)
        x = self.layers(x)
        x = x.permute(0, 2, 1)  # Change shape back to (batch_size, seq_len, hidden_size)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        
        # Define the encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            current_dim = hidden_dim
        encoder_layers.append(nn.Linear(current_dim, encoding_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Define the decoder
        decoder_layers = []
        current_dim = encoding_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            current_dim = hidden_dim
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.init_weights()

    def init_weights(self):
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
