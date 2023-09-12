'''Investigates the data for training and testing'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_utils import *
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import logging
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

parser = argparse.ArgumentParser(description='Investigates the data for training and testing')
parser.add_argument("-l", "--log", default="info",
                    help=("Provide logging level. Example --log debug', default='info'"))
parser.add_argument('stock', type=str, help='Which stock to use (TSLA, MSFT)')
parser.add_argument('-N', '--N_days', type=int, help='Number of the day to consider')

args = parser.parse_args()
levels = {'critical': logging.CRITICAL,
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG}
os.remove('explore')
logging.basicConfig(filename=f'explore', format='%(message)s', level=levels[args.log])

# Load the data
stock = args.stock
if stock == 'TSLA':
    date = '2015-01-01_2015-01-31_10'
    f = 12
elif stock == 'MSFT':
    date = '2018-04-01_2018-04-30_5'
    f = 38

N = args.N_days
levels = 12

# Read the orderbook dataframes
dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
dataframes_paths = [path for path in dataframes_paths if 'orderbook' in path]
dataframes_paths.sort()
dataframes = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in dataframes_paths][:N]
m = 1000

for day in range(N):
    data = dataframes[day].values[:,:levels]
    logging.info(f'Original data:\nShape -> {data.shape}')
    # Convert data entries into int32 to avoid memory issues
    data = data.astype(np.int32)
    logging.info(f'Data sampled every {f} entries:\nShape -> {data[::f].shape}')

    logging.info('\nEvaluate the returns, the volatility, the imbalance and the bid-ask spread...')
    p_a, p_b = data[m:,0][::f], data[m:,2][::f]
    logging.info(f'p_a shape:\n {p_a.shape}')
    logging.info(f'p_b shape:\n {p_b.shape}')

    returns_a = np.diff(p_a)
    returns_a = np.insert(returns_a, 0, 0)
    logging.info(f'Returns shape (zero added at the start):\n {returns_a.shape}')
    returns_b = np.diff(p_b)
    returns_b = np.insert(returns_b, 0, 0)
    logging.info(f'Returns shape (zero added at the start):\n {returns_b.shape}')

    volatility_a = rolling_volatility(data[:,0], m)[::f]
    logging.info(f'Volatility shape:\n {volatility_a.shape}')
    logging.info(f'Volatility Ask: {volatility_a[:20]}')
    volatility_b = rolling_volatility(data[:,2], m)[::f]
    logging.info(f'Volatility shape:\n {volatility_b.shape}')
    logging.info(f'Volatility Bid: {volatility_b[:20]}')

    imbalance = evaluate_imbalance(data, m)[::f]
    logging.info(f'Imbalance shape (zero added at the start):\n {imbalance.shape}')

    spread = evaluate_spread(p_b, p_a)
    logging.info(f'Spread shape:\n {spread.shape}')
    logging.info('Done.')

    # Create 4 subplots and plot the features
    fig, axs = plt.subplots(6, 1, figsize=(10, 10), tight_layout=True)
    n = int(data.shape[0]*0.05)
    axs[0].plot(returns_a[:n])
    axs[0].set_title('Returns Ask')
    axs[1].plot(returns_b[:n])
    axs[1].set_title('Returns Bid')
    axs[2].plot(imbalance[:n])
    axs[2].set_title('Imbalance')
    axs[3].plot(spread[:n])
    axs[3].set_title('Spread')
    axs[4].plot(volatility_a[:n])
    axs[4].set_title('Volatility Ask')
    axs[5].plot(volatility_b[:n])
    axs[5].set_title('Volatility Bid')
    plt.savefig('plots/features.png')

    # Create a new array to store the features
    data = np.empty(shape=(data[m:][::f].shape[0], 6))
    data[:,0] = returns_a
    data[:,1] = returns_b
    data[:,2] = imbalance
    data[:,3] = spread
    data[:,4] = volatility_a
    data[:,5] = volatility_b

    # Create a dictionary to store the features
    features = {'Returns Ask': returns_a, 'Returns Bid': returns_b, 'Imbalance': imbalance, 'Spread': spread, 'Volatility Ask': volatility_a, 'Volatility Bid': volatility_b}
    # Store the dictionary in a dataframe where the columns are the features
    df = pd.DataFrame(features)
    logging.info(f'Dataframe:\n {df.head()}')
    # Save the dataframe
    df.to_parquet(f'../data/{stock}_{date}/features_{day}.parquet')


    # Perform the Ljung-Box test on the features using lags equal to [100, 200, 300 ... 1000]
    logging.info('Perform the Ljung-Box test on the features using lags equal to [100, 200, 300 ... 1000]...')
    logging.info('H0 -> The data are not serially correlated')
    logging.info('H1 -> The data are serially correlated')
    lags = [1000*i for i in range(1, 31)]
    features = ['Returns Ask', 'Returns Bid', 'Imbalance', 'Spread', 'Volatility Ask', 'Volatility Bid']
    for f, i in zip(features, range(data.shape[1])):
        logging.info(f'---------- Feature: {f} ----------')
        result = acorr_ljungbox(data[:n,i], lags=lags)
        for lag, p in zip(lags, result['lb_pvalue']):
            if p < 0.05:
                logging.info(f'{lag} | p-value: {p} | The null hypothesis is rejected (there is serial correlation).')
            else:
                logging.info(f'{lag} | p-value: {p} | The null hypothesis cannot be rejected (there is no serial correlation).')
        logging.info('--------------------------------')
    logging.info('Done.')

    # Plot the autocorrelation of the features
    logging.info('Plot the autocorrelation of the features...')
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), tight_layout=True)
    for feature, i in zip(features,range(data.shape[1])):
        plot_acf(data[:,i], lags=1500, ax=axs[i])
        axs[i].set_title(f'Autocorrelation of {feature}')
    plt.savefig('plots/autocorrelation.png')
    logging.info('Done.')

    # Perform the ADF test on the features
    # logging.info('Perform the ADF test on the features...')
    # for i in range(data.shape[1]):
    #     result = adfuller(data[:,i])
    #     logging.info(f'ADF Statistic: {result[0]}')
    #     logging.info(f'p-value: {result[1]}')
    #     logging.info('Critical Values:')
    #     for key, value in result[4].items():
    #         logging.info(f'\t{key}: {value}')
    # logging.info('Done.')

    # plt.show()

