#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 18:47:10 2023

@author: yogeshsiyyadri
"""

# Import necessary libraries
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set up visualization style
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Create an empty DataFrame to store stock data
df = pd.DataFrame()

for stock in tech_list:
    # Download stock data for each company
    stock_data = yf.download(stock, start, end)
    
    # Add a column for the company name
    stock_data['Company'] = stock
    
    # Concatenate the data to the main DataFrame
    df = pd.concat([df, stock_data])

# Reset the index for clarity
df.reset_index(inplace=True)

# Calculate moving averages and store them in the DataFrame
ma_day = [10, 20, 50, 100]

for ma in ma_day:
    for company in tech_list:
        column_name = f"MA for {ma} days"
        df.loc[df['Company'] == company, column_name] = df.loc[df['Company'] == company, 'Adj Close'].rolling(ma).mean()

# We'll use pct_change to find the percent change for each day
for company in tech_list:
    df.loc[df['Company'] == company, 'Daily Return'] = df.loc[df['Company'] == company, 'Adj Close'].pct_change()

# Plotting
plt.figure(figsize=(15, 10))

# Plot historical view of the closing price
for i, company in enumerate(tech_list, 1):
    plt.subplot(2, 2, i)
    sns.lineplot(data=df[df['Company'] == company], x='Date', y='Adj Close')
    plt.ylabel('Adj Close')
    plt.xlabel(None)
    plt.title(f"Closing Price of {company}")

plt.tight_layout()
plt.show()

# Plotting the total volume of stock being traded each day
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(tech_list, 1):
    plt.subplot(2, 2, i)
    sns.lineplot(data=df[df['Company'] == company], x='Date', y='Volume')
    plt.ylabel('Volume')
    plt.xlabel(None)
    plt.title(f"Sales Volume for {company}")

plt.tight_layout()
plt.show()

# Moving Averages Plot using Seaborn
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

for ma in ma_day:
    plt.subplot(2, 2, ma_day.index(ma) + 1)
    sns.lineplot(data=df, x='Date', y='Adj Close', hue='Company')
    sns.lineplot(data=df, x='Date', y=f'MA for {ma} days', hue='Company', linestyle='dashed')
    plt.title(f"Moving Average for {ma} days")

plt.tight_layout()
plt.show()

# Daily Return Plot using Seaborn
plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")

for company in tech_list:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df[df['Company'] == company], x='Date', y='Daily Return', label=company, linestyle='--', marker='o')
    plt.title(f'Daily Return Percentage for {company}')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.show()

# Histograms of Daily Returns
plt.figure(figsize=(15, 12))

for i, company in enumerate(tech_list, 1):
    df[df['Company'] == company]['Daily Return'].hist(bins=50, alpha=0.75, label=company)
    plt.xlabel('Daily Return')
    plt.ylabel('Counts')
    plt.title(f'{company}')
    plt.legend()

plt.tight_layout()
plt.show()

# Correlation Matrix of Daily Returns
correlation_matrix = df.pivot_table(index='Date', columns='Company', values='Daily Return').corr()

# Display the correlation matrix
print("Correlation Matrix of Daily Returns:")
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', linewidths=.5)
plt.title('Correlation Matrix of Daily Returns')
plt.show()

# Linear Regression for Pairs of Stocks
# Create a list of all possible pairs of stocks
stock_pairs = [('AAPL', 'GOOG'), ('AAPL', 'MSFT'), ('AAPL', 'AMZN'),
               ('GOOG', 'MSFT'), ('GOOG', 'AMZN'), ('MSFT', 'AMZN')]

# Perform linear regression for each stock pair
for stock1, stock2 in stock_pairs:
    # Select the daily returns for the chosen stocks
    returns_df = df[df['Company'].isin([stock1, stock2])][['Date', 'Daily Return', 'Company']]

    # Pivot the DataFrame to have separate columns for each stock's daily returns
    returns_pivot = returns_df.pivot(index='Date', columns='Company', values='Daily Return')

    # Drop missing values
    returns_pivot = returns_pivot.dropna()

    # Ensure numeric data types
    returns_pivot = returns_pivot.apply(pd.to_numeric, errors='coerce')

    # Add a constant term to the independent variable (the daily return of stock2)
    X = sm.add_constant(returns_pivot[stock2])

    # Dependent variable (the daily return of stock1)
    y = returns_pivot[stock1]

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Print the regression results
    print(f"\nLinear Regression Results for {stock1} vs {stock2}:\n")
    print(model.summary())
