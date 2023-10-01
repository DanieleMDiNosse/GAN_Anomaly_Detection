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
from multiprocessing import Pool


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
        # Rename the columns
        if 'message' in path:
            # if len(data.columns) > 6: eliminate the last column
            if len(data.columns) > 6: data = data.drop(columns = data.columns[-1])
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


def evaluate_imbalance(data, f):
    '''This function evaluates the imbalance between the bid and ask sides of the order book up to a certain level.
    It averages over a number of events equal to the sampling frequency.
    The imbalance is defined as the ratio between the bid volumes and the sum of the bid and ask volumes.
    
    Parameters
    ----------
    data : numpy array
        Array containing the order book data.
    f : int
        Sampling frequency.

    Returns
    -------
    imbalance : numpy array
        Array containing the imbalance for each time step.'''

    volumes = data[:,[i for i in range(1, data.shape[1], 2)]]
    v_a = volumes[:, ::2]
    v_b = volumes[:, 1::2]
    v_b_summed = np.array([v_b[i:i+f].sum(axis=0) for i in range(0, v_b.shape[0]-f)])
    v_a_summed = np.array([v_a[i:i+f].sum(axis=0) for i in range(0, v_a.shape[0]-f)])
    imbalance = v_b_summed.sum(axis=1) / (v_b_summed.sum(axis=1) + v_a_summed.sum(axis=1))
    return imbalance

def rolling_volatility(p, m):
    '''This function computes the rolling volatility of the ask prices up to a certain level.
    It averages over a number of events equal to the m.
    The rolling volatility is defined as the standard deviation of the returns.
    
    Parameters
    ----------
    p_a : numpy array
        Ask prices.
    m : int
        Size of the window.
    
    Returns
    -------
    rolling_volatility : numpy array
        Array containing the rolling volatility for each time step.'''
    
    returns = np.diff(p)
    returns = np.insert(returns, 0, 0)
    rolling_volatility = np.array([returns[i:i+m].std() for i in range(0, returns.shape[0]-m)])
    return rolling_volatility

def evaluate_spread(p_b, p_a):
    '''This function evaluates the spread between the bid and ask sides of the order book.
    The spread is defined as the difference between the best ask price and the best bid price.
    
    Parameters
    ----------
    p_b : numpy array
        Bid prices.
    p_a : numpy array
        Ask prices.
        
    Returns
    -------
    spread : numpy array
        Array containing the spread for each time step.'''
    
    spread = p_a - p_b
    return spread

def input_generation(returns, imbalance, spread, window_size, condition_size):
    '''This function generates the input for the GAN. The input is a multivariate time series
    composed by the returns, the imbalance and the spread. The function returns a numpy array
    containing the input data.
    
    Parameters
    ----------
    returns : numpy array
        Array containing the returns.
    imbalance : numpy array
        Array containing the imbalance.
    spread : numpy array
        Array containing the spread.
    window_size : int
        Length of the sliding window.
    condition_size : int
        Length of the condition.
    
    Returns
    -------
    input_data : numpy array
        Array containing the input data.'''
    
    input_data = np.zeros((returns.shape[0] - window_size, window_size, returns.shape[1] + imbalance.shape[1] + spread.shape[1]))
    for i in range(returns.shape[0] - window_size):
        input_data[i, :, :returns.shape[1]] = returns[i:i+window_size, :]
        input_data[i, :, returns.shape[1]:returns.shape[1]+imbalance.shape[1]] = imbalance[i:i+window_size, :]
        input_data[i, :, returns.shape[1]+imbalance.shape[1]:] = spread[i:i+window_size, :]
    return input_data


@nb.jit(nopython=True)
def divide_data_condition_input(data, condition_size):
    '''Divide the data into condition and input data. The condition data is the data that
    is used to condition the GAN
    
    Parameters
    ----------
    data : numpy array or list
        Array containing the data (like training data for instance).
    condition_size : int
        Length of the condition.
    
    Returns
    -------
    condition : list
        Array containing the condition data.
    input_data : list
        Array containing the input data.'''
    
    condition = []
    input_data = []
    for window in data:
        condition.append(window[:condition_size, :])
        input_data.append(window[condition_size:, :])

    return condition, input_data

def parallel_divide_data(data, condition_size):
    num_workers = 4
    partial_size = len(data) // num_workers
    with Pool(num_workers) as p:
        results = p.starmap(divide_data_condition_input, [(data[i*partial_size : (i+1)*partial_size], condition_size) for i in range(num_workers)])
    
    condition = np.concatenate([result[0] for result in results])
    input_data = np.concatenate([result[1] for result in results])
    return condition, input_data

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
    timedeltas : list
        List of the time intervals (real time) for each window size across all the dataset.
    '''

    times = data['Time'].values
    #times = times / 1e9 # Convert to seconds
    step = 100
    window_size = [step*i for i in range(1, 21)]
    timedeltas = []

    for i in range(len(window_size)):
        # Use the numba implementation for speedup
        timedeltas.append(sliding_windows_stat_numba(times, nb.typed.List(window_size), i))

    return timedeltas

@nb.jit(nopython=True)
def sliding_windows_stat_numba(times, window_size, i):
    timedeltas = []
    for j in range(times.shape[0] - window_size[i]):
        timedelta = times[window_size[i] + j] - times[j]
        timedeltas.append(timedelta/60) # Convert to minutes each timedelta
    
    return timedeltas

@nb.jit(nopython=True)
def divide_vector(vector, N):
    '''This function takes as input a vector and divides it into N subvectors of the same
    length. The last subvector can have a different length.
    
    Parameters
    ----------
    vector : numpy array
        Vector to be divided.
    N : int
        Number of subvectors.
    
    Returns
    -------
    subvectors : list
        List of subvectors.
    '''
    subvectors = []
    length = int(vector.shape[0]/N)
    for i in range(N):
        subvectors.append(vector[i*length:(i+1)*length])
    subvectors.append(vector[(i+1)*length:])
    return subvectors

def fast_autocorrelation(x, alpha=0.05):
    n = len(x)
    x = np.pad(x, (0, n), mode='constant')  # Zero padding
    f = np.fft.fft(x)
    p = np.absolute(f)**2
    r = np.fft.ifft(p)
    r = np.real(r)[:n]
    r = r / np.max(r)  # Normalize
    lag_no_corr = np.where(r < alpha)[0][0]

    return r, lag_no_corr

def preprocessing_message_df(message_df, m=1000):
    # Drop columns Order ID
    message_df = message_df.drop(columns=['Order ID'])
    # Filter by Event type equal to 1,2,3 or 4
    message_df = message_df[message_df['Event type'].isin([1,2,3,4])]
    # Take the rows from the f-th one
    message_df = message_df.iloc[m:]

    return message_df, message_df.index

@nb.jit(nopython=True)
def divide_into_overlapping_pieces(data, overlap_size, num_pieces):
    """
    Divide a vector into a number of overlapping pieces.

    Parameters
    ----------
    data : array_like
        The data to be divided.
    overlap_size : int
        The number of overlapping samples between two consecutive pieces.
    num_pieces : int
        The number of pieces to divide the data into.

    Returns
    -------
    pieces : list of array_like
        The divided pieces.
    """
    piece_size = int(len(data) / num_pieces + (1 - 1/num_pieces) * overlap_size)
    pieces = []
    for i in range(num_pieces):
        start = i * (piece_size - overlap_size)
        end = start + piece_size
        if i == num_pieces - 1:
            end = len(data)
        pieces.append(data[start:end-1])
    return pieces


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

    # Check if all the first columns names of the dataframes are equal to 'Time'.
    # If not, rename them via the rename_columns function.
    for path, data in zip(dataframes_paths, dataframes):
        if 'message' in path:
            if data.columns[0] != 'Time':
                logging.info('Renaming the columns of the dataframes...')
                rename_columns(stock, date)
        elif 'orderbook' in path:
            if data.columns[0] != 'Ask price 1':
                logging.info('Renaming the columns of the dataframes...')
                rename_columns(stock, date)


    if args.box_plot_sw:
        '''The window size of the sliding window is a hyperparameter that needs to be tuned. Here I plot the boxplots
         of the time intervals for each window size length.'''
        i = 0
        for data in tqdm(dataframes, desc='BoxPlots of delta_t for each day'):
            i += 1
            if list(data.columns) == ['Time', 'Event type', 'Order ID', 'Size', 'Price', 'Direction']:
                timedeltas = sliding_windows_stat(data)
                # print(timedeltas)
                plt.figure(figsize=(10, 5), tight_layout=True)
                plt.boxplot(timedeltas)
                plt.xticks(np.arange(1, len(window_size)+1), window_size, rotation=45)
                plt.xlabel('Window size')
                plt.ylabel('Time interval (minutes)')
            plt.title(f'Time intervals for each window size - {stock}')
            plt.savefig(f'plots/time_intervals_{stock}_{i}.png')
        plt.show()
        exit()

    

