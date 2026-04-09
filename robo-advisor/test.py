import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta  # Technical Analysis

class TradingAdvisor:
    def __init__(self, stock_symbol):
        self.stock_symbol = stock_symbol
        self.data = None

    def fetch_data(self, start_date, end_date):
        self.data = yf.download(self.stock_symbol, start=start_date, end=end_date)
        self.data = self.data[['Close']]

    def calculate_indicators(self):
        self.data['SMA_20'] = ta.trend.sma_indicator(self.data['Close'], window=20)
        self.data['SMA_50'] = ta.trend.sma_indicator(self.data['Close'], window=50)
        self.data['RSI'] = ta.momentum.rsi(self.data['Close'], window=14)

    def generate_signals(self):
        self.data['Signal'] = 0
        self.data['Signal'][20:] = np.where(
            (self.data['SMA_20'][20:] > self.data['SMA_50'][20:]), 1, 0
        )
        self.data['Position'] = self.data['Signal'].diff()

    def plot_data(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Close'], label='Close Price', alpha=0.5)
        plt.plot(self.data['SMA_20'], label='20-Day SMA', linestyle='--')
        plt.plot(self.data['SMA_50'], label='50-Day SMA', linestyle='--')

        plt.plot(self.data[self.data['Position'] == 1].index,
                 self.data['SMA_20'][self.data['Position'] == 1],
                 '^', markersize=10, color='g', label='Buy Signal')

        plt.plot(self.data[self.data['Position'] == -1].index,
                 self.data['SMA_20'][self.data['Position'] == -1],
                 'v', markersize=10, color='r', label='Sell Signal')

        plt.title(f'Trading Signals for {self.stock_symbol}')
        plt.legend()
        plt.show()

    def run(self, start_date, end_date):
        self.fetch_data(start_date, end_date)
        self.calculate_indicators()
        self.generate_signals()
        self.plot_data()

# Example Usage
advisor = TradingAdvisor('AAPL')  # Replace 'AAPL' with the desired stock
advisor.run('2022-01-01', '2023-01-01')

