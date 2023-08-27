'''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
composed by a sliding window that shifts by one time step each time.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import logging
import numba as nb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def rename_files(folder_path):
    old_names = os.listdir(folder_path)
    old_names_splitted = [name.split('_') for name in old_names]
    new_names = [f'{name[0]}_{name[1]}_{name[4]}.parquet' for name in old_names_splitted]
    for old_name, new_name in zip(old_names, new_names):
        os.rename(f'{folder_path}/{old_name}', f'{folder_path}/{new_name}')
    return None

def rename_columns(stock, date):
    # Read the message dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    # dataframes_paths = [path for path in dataframes_paths if 'orderbook' in path]
    dataframes_paths.sort()
    dataframes = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in dataframes_paths]
    for data, path in zip(dataframes, dataframes_paths):
        print(path)
        # Rename the columns
        if 'message' in path:
            data.columns = ['Time', 'Event type', 'Order ID', 'Size', 'Price', 'Direction']
        elif 'orderbook' in path:
            n = data.shape[1]
            ask_price_columns = [f'Ask price {i}' for i,j in zip(range(1, int(n/2)+1), range(0,n,4))]
            ask_size_columns = [f'Ask size {i}' for i,j in zip(range(1, int(n/2)+1), range(1,n,4))]
            bid_price_columns = [f'Bid price {i}' for i,j in zip(range(1, int(n/2)+1), range(2,n,4))]
            bid_size_columns = [f'Bid size {i}' for i,j in zip(range(1, int(n/2)+1), range(3,n,4))]
            ask_columns = [[ask_price_columns[i], ask_size_columns[i]] for i in range(len(ask_size_columns))]
            bid_columns = [[bid_price_columns[i], bid_size_columns[i]] for i in range(len(bid_size_columns))]
            columns = np.array([[ask_columns[i], bid_columns[i]] for i in range(len(ask_size_columns))]).flatten()
            data.columns = columns
        data.to_parquet(f'../data/{stock}_{date}/{path}')
    return None

@nb.jit(nopython=True)
def divide_into_windows(data, window_size):
    """Divide the time series into windows of length window_size, each shifted by one time step."""
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size, :])
    return windows

def sliding_windows_stat(data):
    '''This function computes the statistics of the time intervals between two consecutive
    windows. It returns a dataframe containing the mean and the standard deviation of the
    time intervals for each window size. It uses another function, sliding_windows_stat_numba,
    which is a numba implementation of the most time consuming part of the function.
    
    Parameters
    ----------
    data : pandas dataframe
        Dataframe containing the message data.
    
    Returns
    -------
    window_sizes : list
        List containing the window sizes.
    minutes_statistics : pandas dataframe
        Dataframe containing the mean and the standard deviation of the time intervals for each window size.'''

    times = data['Time'].values
    #times = times / 1e9 # Convert to seconds
    step = 100
    window_size = [step*i for i in range(1, 21)]
    timedeltas = []

    for i in tqdm(range(len(window_size))):
        # Use the numba implementation for speedup
        timedeltas.append(sliding_windows_stat_numba(times, window_size, i))
        # Compute the mean and the standard deviation of the time intervals
        # timedelta_mean = np.mean(timedeltas)
        # total_minutes_mean[i] = timedelta_mean
        # total_minutes_bounds[0,i] = min(timedeltas)
        # total_minutes_bounds[1,i] = max(timedeltas)
    
    # minutes_statistics = pd.DataFrame({'window_size': window_size, 'mean': total_minutes_mean, 'min': total_minutes_bounds[0], 'max': total_minutes_bounds[1]})
    # minutes_statistics.to_pickle(f'../data/minutes_statistics_{os.getpid()}.pkl')
    return timedeltas

@nb.jit(nopython=True)
def sliding_windows_stat_numba(times, window_size, i):
    timedeltas = []
    for j in range(times.shape[0] - window_size[i]):
        timedelta = times[window_size[i] + j] - times[j]
        timedeltas.append(timedelta/60) # Convert to minutes each timedelta
    
    return timedeltas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-r', '--rename', action='store_true', help='Rename the colums of the dafaframes')
    parser.add_argument('stock', type=str, help='Which stock to use (TSLA, MSFT)')
    parser.add_argument('-N', '--N_days', type=int, help='Number of the day to consider')
    parser.add_argument('-bpsw', '--box_plot_sw', action='store_true', help='Box plots for the time intervals within each window size')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    logging.basicConfig(level=levels[args.log])

    # Load the data
    stock = args.stock
    if stock == 'TSLA':
        date = '2015-01-01_2015-01-31_10'
    elif stock == 'MSFT':
        date = '2018-04-01_2018-04-30_5'

    N = args.N_days
    
    # Read the message dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    # dataframes_paths = [path for path in dataframes_paths if 'orderbook' in path]
    dataframes_paths.sort()
    dataframes = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in dataframes_paths][:N]
    step = 100
    window_size = [step*i for i in range(1, 21)]

    if args.rename:
        pass

    if args.box_plot_sw:
        '''The window size of the sliding window is a hyperparameter that needs to be tuned. Here I plot the boxplots
         of the time intervals for each window size length.'''

        for data in tqdm(dataframes):
            window_size, timedeltas = sliding_windows_stat(data)
            # Concatenate the dataframe that sliding_windows_stats returns with the previous one
            plt.figure(figsize=(10, 5), tight_layout=True)
            plt.boxplot(timedeltas)
            # window_size.insert(0, 0)
            plt.xticks(np.arange(1, len(window_size)+1), window_size, rotation=45)
            plt.xlabel('Window size')
            plt.ylabel('Time interval (minutes)')
        plt.show()

    # For now consider just one window size and one day
    window_size = 500
    data = dataframes[0].values
    input_data = np.array(divide_into_windows(data, window_size))

    # Normalize all the features with StandardScaler
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data.reshape(-1, input_data.shape[-1])).reshape(input_data.shape)
    
    # Divide the data into train, validation and test set
    train, test = train_test_split(input_data, test_size=0.2, shuffle=False)
    train, val = train_test_split(train, test_size=0.2, shuffle=False)
