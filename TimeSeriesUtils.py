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
