import pandas as pd
import numpy as np

# Load the data
file_path = 'final_data.csv'

stock_data = pd.read_csv(file_path)

# Check for and drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in stock_data.columns:
    stock_data.drop(columns=['Unnamed: 0'], inplace=True)

# Convert 'Outstanding_Shares' to numeric (if not already)
stock_data['Outstanding_Shares'] = pd.to_numeric(stock_data['Outstanding_Shares'], errors='coerce')

# Multiply 'Outstanding_Shares' by 10,000,000 (since it's in crores) before calculating turnover
stock_data['Outstanding_Shares'] *= 10000000

# Calculate daily turnover
stock_data['Turnover'] = stock_data['Volume'] / stock_data['Outstanding_Shares']

# Convert 'Date' to datetime format
stock_data['Date'] = pd.to_datetime(stock_data['Date'], format="%d/%m/%Y")

# Set 'Date' as the index
stock_data.set_index('Date', inplace=True)

# Resample to monthly frequency to get the first price of each month
monthly_data = stock_data.groupby('Symbol').resample('BMS').first()

# Reset the index to make 'Symbol' and 'Date' columns again
monthly_data = monthly_data.reset_index(level=0, drop=True)

# Calculate monthly returns
monthly_data['Monthly_Return'] = monthly_data.groupby('Symbol')['Adj Close'].pct_change()

# Calculate cumulative returns for 3, 6, 9, and 12 months
monthly_data['3M_Return'] = monthly_data.groupby('Symbol')['Adj Close'].pct_change(3)
monthly_data['6M_Return'] = monthly_data.groupby('Symbol')['Adj Close'].pct_change(6)
monthly_data['9M_Return'] = monthly_data.groupby('Symbol')['Adj Close'].pct_change(9)
monthly_data['12M_Return'] = monthly_data.groupby('Symbol')['Adj Close'].pct_change(12)

monthly_data.reset_index(inplace=True)

# Rank stocks based on cumulative returns
for period in ['3M', '6M', '9M', '12M']:
    rank_col = f'{period}_Rank'
    monthly_data[rank_col] = monthly_data.groupby('Date')[f'{period}_Return'].rank(method='first', ascending=False)

# Define quintile function with check for sufficient unique values
def assign_quintile(x, rank_col, quintile_col):
    if len(x[rank_col].unique()) < 10:
        x[quintile_col] = np.nan
    else:
        x[quintile_col] = pd.qcut(x[rank_col], 10, labels=False) + 1
    return x

# Apply quintile ranking
for period in ['3M', '6M', '9M', '12M']:
    rank_col = f'{period}_Rank'
    quintile_col = f'{period}_Quintile'
    monthly_data = monthly_data.groupby('Date').apply(assign_quintile, rank_col, quintile_col).reset_index(drop=True)


# Calculate average turnover over the formation periods
for period in ['3M', '6M', '9M', '12M']:
    avg_turnover_col = f'{period}_Avg_Turnover'
    window_size = int(period[:-1])
    monthly_data[avg_turnover_col] = monthly_data.groupby('Symbol')['Turnover'].rolling(window=window_size,min_periods=1).mean().reset_index(level=0, drop=True)


# Rank stocks based on average turnover
for period in ['3M', '6M', '9M', '12M']:
    avg_turnover_col = f'{period}_Avg_Turnover'
    rank_col = f'{period}_Turnover_Rank'
    monthly_data[rank_col] = monthly_data.groupby('Date')[avg_turnover_col].rank(method='first')

# Define tercile function with check for sufficient unique values
def assign_tercile(x, rank_col, tercile_col):
    if len(x[rank_col].unique()) < 3:
        x[tercile_col] = np.nan
    else:
        x[tercile_col] = pd.qcut(x[rank_col], 3, labels=False) + 1
    return x

# Apply tercile ranking
for period in ['3M', '6M', '9M', '12M']:
    rank_col = f'{period}_Turnover_Rank'
    tercile_col = f'{period}_Turnover_Tercile'
    monthly_data = monthly_data.groupby('Date').apply(assign_tercile, rank_col, tercile_col).reset_index(drop=True)
