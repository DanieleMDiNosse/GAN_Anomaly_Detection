'''This file contains the metrics used to evaluate the performance of the CWGAN model.'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import os

def return_distribution(fake_samples):
    '''Returns the distribution of the generated samples together with
    the corresponding autocorrelation function.'''

    ask_price = (fake_samples[:, 0] / 100).astype(int)
    returns = np.diff(ask_price)
    value, count = np.unique(returns, return_counts=True)
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.bar(value, count)
    plt.title('Return Distribution')
    # x_ticks = np.arange(value.min(), value.max(), 1)
    plt.savefig(f'plots/{os.getpid()}/return_distribution.png')
    
    # use plot_acf and save the autocorrelation plot
    plot_acf(returns, lags=100)
    plt.savefig(f'plots/{os.getpid()}/autocorrelation.png')
    pass

def volatility(fake_samples):
    '''Returns the volatility of the generated samples. It is known that
    the volatility is characterized by clustering.'''
    ask_price = (fake_samples[:, 0] / 100).astype(int)
    bid_price = (fake_samples[:, 1] / 100).astype(int)
    mid_price = (ask_price + bid_price) / 2
    returns = np.diff(mid_price)
    volatility = np.std(returns)
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(volatility)
    plt.title('Volatility')
    plt.savefig(f'plots/{os.getpid()}/volatility.png')
    pass

def bid_ask_spread(fake_samples):
    '''Returns the bid-ask spread of the generated samples.'''
    ask_price = (fake_samples[:, 0] / 100).astype(int)
    bid_price = (fake_samples[:, 1] / 100).astype(int)
    spread = ask_price - bid_price
    value, count = np.unique(spread, return_counts=True)
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.bar(value, count)
    plt.title('Bid-Ask Spread Distribution')
    plt.savefig(f'plots/{os.getpid()}/bid_ask_spread.png')
    pass