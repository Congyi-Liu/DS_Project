# Converted from Jupyter Notebook

import pandas as pd
import numpy as np
import datetime as dt
import random
import re
import time

# Data
import yfinance as yf
from full_fred.fred import Fred

# ML
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler

fred = Fred('fred_Key.txt')
fred.set_api_key_file('fred_Key.txt')

# Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to calculate percent change for Yahoo Finance data (open/close-1)
def calculate_percent_change_yf(df):
    return ((df['Close'] - df['Open']) / df['Open'])

# Function to calculate percent change for other data (current/previous-1)
def calculate_percent_change_generic(df):
    df = pd.to_numeric(df)
    return df.pct_change()

# Data Importing
print('Loading Data...')
begin_date = '2023-01-01'
end_date = '2024-01-01' 

# S&P 500 Index price
spy_df = yf.download('^GSPC', start=begin_date, end=end_date)

# VIX (Volatility Index)
vix_df = yf.download('^VIX', start=begin_date, end=end_date)

# FRED Data
# Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL)
df_CPIAUCSL = fred.get_series_df('CPIAUCSL')
df_CPIAUCSL.index = pd.to_datetime(df_CPIAUCSL['date'])

# Real Gross Domestic Product (GDP)
df_GDP = fred.get_series_df('GDP')
df_GDP.index = pd.to_datetime(df_GDP['date'])
df_GDP['value'] = pd.to_numeric(df_GDP['value'], errors='coerce')
df_GDP['value'].ffill(inplace=True)

# Unemployment Rate (UNRATE)
df_UNRATE = fred.get_series_df('UNRATE')
df_UNRATE.index = pd.to_datetime(df_UNRATE['date'])

print("\nAll Data Loaded")

# Applying percent change for Yahoo Finance data
vix_df['Percent Change'] = calculate_percent_change_yf(vix_df)
spy_df['Percent Change'] = calculate_percent_change_yf(spy_df)

# Applying percent change for FRED data
df_CPIAUCSL['Percent Change'] = calculate_percent_change_generic(df_CPIAUCSL['value'])
df_GDP['Percent Change'] = calculate_percent_change_generic(df_GDP['value'])
df_UNRATE['Percent Change'] = calculate_percent_change_generic(df_UNRATE['value'])

print("Percent changes for all data calculated.")

# Forward VIX for Forecasting
vix_for = pd.DataFrame(vix_df)
vix_for = vix_for.Close.shift(-1)

# Data Cleaning
# Use outer join to combine columns
train = pd.concat([vix_for, 
                   vix_df['Percent Change'], 
                   spy_df['Percent Change'],
                   df_CPIAUCSL['Percent Change'],
                   df_GDP['Percent Change'],
                   df_UNRATE['Percent Change']], axis=1, join='outer')

var = ['vix_for', 'vix', 'spy', 'CPI', 'GDP', 'UNRATE']
train.columns = var

# Forward Fill NA with periodical rates
train[['CPI', 'GDP', 'UNRATE']] = train[['CPI', 'GDP', 'UNRATE']].ffill()

X_pred = train.drop(['vix_for'], axis=1)
X_pred = X_pred.dropna()
X_pred = X_pred.tail(1)

train.replace([np.inf, -np.inf], 0, inplace=True)
train = train.dropna()

# Separate X and Y
y = np.array(train['vix_for'])
X = np.array(train.drop(['vix_for'], axis=1))

# Apply scaling
scaler = StandardScaler()
# Fit and transform the training data
X = scaler.fit_transform(X)

# Separate Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Initialize models
ols_model = LinearRegression()
elastic_net_model = ElasticNet(random_state=42)
random_forest_model = RandomForestRegressor(random_state=42)
neural_network_model = MLPRegressor(random_state=42, max_iter=500)

# Train and evaluate OLS model
ols_model.fit(X_train, y_train)
y_pred_ols = ols_model.predict(X_test)
mae_ols = mean_absolute_error(y_test, y_pred_ols)

# Train and evaluate Elastic Net model
elastic_net_model.fit(X_train, y_train)
y_pred_elastic_net = elastic_net_model.predict(X_test)
mae_elastic_net = mean_absolute_error(y_test, y_pred_elastic_net)

# Train and evaluate Random Forest model
random_forest_model.fit(X_train, y_train)
y_pred_random_forest = random_forest_model.predict(X_test)
mae_random_forest = mean_absolute_error(y_test, y_pred_random_forest)

# Train and evaluate Neural Network model
neural_network_model.fit(X_train, y_train)
y_pred_neural_network = neural_network_model.predict(X_test)
mae_neural_network = mean_absolute_error(y_test, y_pred_neural_network)

# Plot Mean Absolute Error for all models
mae_values = [mae_ols, mae_elastic_net, mae_random_forest, mae_neural_network]
model_names = ['OLS', 'Elastic Net', 'Random Forest', 'Neural Network']
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6), dpi=300)
plt.bar(model_names, mae_values, color='skyblue')
plt.xlabel('Machine Learning Model')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE Comparison of Different Machine Learning Models')
plt.show()