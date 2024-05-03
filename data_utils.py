'''This script is aimed to preprocess the data for model training.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import logging
import numba as nb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from PIL import ImageDraw
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest, ttest_ind
import math
from sklearn.decomposition import PCA
import seaborn as sns

def convert_and_renamecols(dataframes_folder_path):
    '''This function takes as input the path of the folder containing the dataframes and renames the columns
    of the message and orderbook dataframes. Then, it saves the dataframes in the same folder as a parquet
    dataframe.
    
    Parameters
    ----------
    dataframes_folder_path : string
        Path of the folder containing the dataframes.
    
    Returns
    -------
    None.'''

    # # Read the message dataframes
    dataframes_paths = os.listdir(f'{dataframes_folder_path}')
    dataframes_paths.sort()
    for df in dataframes_paths:
        path = f'{dataframes_folder_path}/{df}'
        logging.info(path)
        if path.endswith('.csv'): 
            logging.info(f'Found csv dataframe for {path}...')
            data = pd.read_csv(f'{path}')
        if path.endswith('.pkl'): 
            logging.info(f'Found pkl dataframe for {path}...')
            data = pd.read_pickle(f'{path}')
        if path.endswith('.parquet'): 
            logging.info(f'Found parquet dataframe for {path}...')
            data = pd.read_parquet(f'{path}')

        # Rename the columns
        if 'message' in path:
            # if len(data.columns) > 6: eliminate the last column
            if len(data.columns) > 6: 
                data = data.drop(columns = data.columns[-1])
            data.columns = ['Time', 'Event type', 'Order ID', 'Size', 'Price', 'Direction']
        elif 'orderbook' in path:
            n = data.shape[1]
            ask_price_columns = [f'Ask price {i}' for i,j in zip(range(0, int(n/2)), range(0,n,4))]
            ask_size_columns = [f'Ask size {i}' for i,j in zip(range(0, int(n/2)), range(1,n,4))]
            bid_price_columns = [f'Bid price {i}' for i,j in zip(range(0, int(n/2)), range(2,n,4))]
            bid_size_columns = [f'Bid size {i}' for i,j in zip(range(0, int(n/2)), range(3,n,4))]
            ask_columns = [[ask_price_columns[i], ask_size_columns[i]] for i in range(len(ask_size_columns))]
            bid_columns = [[bid_price_columns[i], bid_size_columns[i]] for i in range(len(bid_size_columns))]
            columns = np.array([[ask_columns[i], bid_columns[i]] for i in range(len(ask_size_columns))]).flatten()
            data.columns = columns

        # Save the dataframe
        name = f"{path.split('/')[-1].split('.')[0]}.parquet"
        folder = path.split('/')[-2]
        logging.info(name)
        data.to_parquet(f'../data/{folder}/{name}')
    return None

def preprocessing_orderbook_df(orderbook, message, sampling_seconds=10, discard_time=1800):
    '''This function takes as input the orderbook dataframe provided by LOBSTER.com and preprocessed via
    convert_and_renamecols and then performs the following operations:
    - Discard a certain number of rows at the beginning and the end of the dataframe, since they are likely "non stationary"
    - Sample the dataframe every 10 seconds
    - Reset the index
    - Drop the columns Order ID and Time
    
    Parameters
    ----------
    orderbook : pandas dataframe
        Orderbook dataframe preprocessed via convert_and_renamecols.
    message : pandas dataframe
        Message dataframe preprocessed via convert_and_renamecols. Used only to take the Time column.
    sampling_seconds : int
        Sampling frequency. The default is 10.
    discard_time : int
        Number of seconds to discard from the beginning and the end of the dataframe. The default is 1800.
    
    Returns
    -------
    orderbook : pandas dataframe
        Preprocessed orderbook dataframe.
    [k, l] : list
        List of two indexes: one corresponds to the 30th minute of the day and the other to the 30-minute-to-end minute.'''

    # Take the Time column of message and add it to orderbook
    orderbook['Time'] = message['Time']

    # Check the Time column and find the index k such that (orderbook['Time'][k] - orderbook['Time'][0])=discard_time.
    # Similar task is performed for the end of the dataframe
    start = orderbook['Time'].values[0]
    end = orderbook['Time'].values[-1]
    for i in range(len(orderbook)):
        if orderbook['Time'].values[i] - start >= discard_time:
    
            k = i
            break
    for i in range(len(orderbook)):
        if end - orderbook['Time'].values[-i] >= discard_time:
            l = len(orderbook) - i
            break
    # Discard the first k and the last l rows
    orderbook = orderbook.iloc[k:l, :]

    # Initialize an empty list to store the indices of the rows to be selected
    selected_indices = []

    # Start with the first row
    current_time = orderbook['Time'].values[0]
    selected_indices.append(0)

    # Iterate through the dataframe to select rows approximately sampling_seconds seconds apart
    for i in range(1, len(orderbook)):
        if orderbook['Time'].values[i] - current_time >= sampling_seconds:
            selected_indices.append(i)
            current_time = orderbook['Time'].values[i]

    # Create a new dataframe with the selected rows
    orderbook = orderbook.iloc[selected_indices]
    orderbook = orderbook.reset_index(drop=True)
    return orderbook, [k, l]

def prices_and_volumes(orderbook):
    '''This function takes as input the orderbook dataframe and returns the bid and ask prices and volumes.
    
    Parameters
    ----------
    orderbook : pandas dataframe
        Orderbook dataframe.
    
    Returns
    -------
    bid_prices, bid_volumes, ask_prices, ask_volumes : numpy array
        Arrays containing prices and volumes.'''

    n = orderbook.shape[1]
    bid_prices = np.array(orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]])
    bid_volumes = np.array(orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]])
    ask_prices = np.array(orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]])
    ask_volumes = np.array(orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]])
    return bid_prices, bid_volumes, ask_prices, ask_volumes

def volumes_per_level(ask_prices, bid_prices, ask_volumes, bid_volumes, depth):
    '''This function takes as input the bid and ask prices and volumes and returns the volumes at each price level
    for the first depth levels (consecutive levels are separated by one tick). Note that not all the levels are necessarily occupied.
    Note also that the bid_volumes are net to be negative.
    
    Parameters
    ----------
    ask_prices : numpy array
        Array containing the ask prices.
    bid_prices : numpy array
        Array containing the bid prices.
    ask_volumes : numpy array
        Array containing the ask volumes.
    bid_volumes : numpy array
        Array containing the bid volumes.
    depth : int
        Depth of the LOB.
        
    Returns
    -------
    volume_ask, volume_bid : numpy array
        Arrays containing the volumes at each price level for the first depth levels.'''

    volume_ask = np.zeros(shape=(ask_prices.shape[0], depth))
    volume_bid = np.zeros(shape=(ask_prices.shape[0], depth))
    for row in range(0, ask_prices.shape[0], 2):
        start_ask = ask_prices[row, 0] # best ask price
        start_bid = bid_prices[row, 0] # best bid price
        volume_ask[row, 0] = ask_volumes[row, 0] # volume at best ask
        volume_bid[row, 0] = -bid_volumes[row, 0] # volume at best bid

        for i in range(1,depth): #from the 2th until the 4th tick price level
            if ask_prices[row, i] == start_ask + 100*i: # check if the next occupied level is one tick ahead
                volume_ask[row, i] = ask_volumes[row, i]
            else:
                volume_ask[row, i] = 0

            if bid_prices[row, i] == start_bid - 100*i: # check if the next occupied level is one tick before
                volume_bid[row, i] = -bid_volumes[row, i]
            else:
                volume_bid[row, i] = 0

    return volume_ask, volume_bid

def LOB_snapshots(orderbook, k, m, n):
    '''This function's goal is to create the appropiate dataframe to be fed into the GAN in order to learn the transition
    probabilities of the LOB of the form p(X(t+1), X(t+2),..., X(t+m)|X(t), X(t-1),...,X(t-n)).
    It takes as input an orderbook dataframe (preprocessed via preprocessing_orderbook_df), the desired 
    depth k of the LOB on both sides, the number of future timestamp m to consider and the number of past timestamps n to set
    as condition.
    
    For instance, if k=3 is the depth of the LOB, the volumes array is composed by 2k+1=7 elements,
    where the last value correspond do the price change in ticks. The length of the volume array is
    2L-2, where L is the length of the orderbook dataframe.
    
    Example
    -------
    Consider the following LOB with k=3 and n,m=1 at time t:
    0   1   2   3   4   5
    |---|---|---|---|---|
    Here 2 and 3 correspond to the best ask and bid prices, respectively.

    At time t+1 either one of the following events can happen:
    1) The best ask and bid prices remain the same (no price change)
    2) The best ask increases (price goes up)
    3) The best bid decreases (price goes down)

    For every timestamp in the LOB, we center the LOB (snapshot at time t) and then we
    consider what happens at the next timestamp (t+1), keeping fixed the price grid.

    Note: here we suppose the spread is always of size equal to one tick.
    '''
    length = orderbook.shape[0]
    # logging.info(f'Length of the original orderbook datafrme:\n\t{length}')
    # the additional column 2k+1 is used to track the price change.
    # The lenght is set to 2*length because I want to learn the transition probability p(X(t+1)|X(t))
    # where X(t+1) becames X(t) at the next time step.
    volumes = np.zeros(shape=(2*length-2, 2*k+1))

    for i in range(0, length-n):
        # Set the price grid for time t and create the LOB snapshot at time t
        best_ask = orderbook['Ask price 0'].values[i]
        best_bid = orderbook['Bid price 0'].values[i]
        volumes[2*i, k] = orderbook['Ask size 0'].values[i]
        volumes[2*i, k-1] = -orderbook['Bid size 0'].values[i]
        volumes[2*i, -1] = 99
        for j in range(1, k):
            # Check if the next price levels are occupied
            if orderbook[f'Ask price {j}'].values[i] == best_ask + 100*j:
                volumes[2*i, k+j] = orderbook[f'Ask size {j}'].values[i]
            else:
                volumes[2*i, k+j] = 0

            if orderbook[f'Bid price {j}'].values[i] == best_bid - 100*j:
                volumes[2*i, k-(j+1)] = -orderbook[f'Bid size {j}'].values[i]
            else:
                volumes[2*i, k-(j+1)] = 0

        # Now check the next timestamp. I consider just the next one because I want to learn the
        # transition probability p(X(t+1)|X(t))

        if orderbook['Ask price 0'].values[i+1] == best_ask:
            volumes[2*i+1, k] = orderbook['Ask size 0'].values[i]
            volumes[2*i+1, k-1] = -orderbook['Bid size 0'].values[i]
            # No price change at t+1. Best bid and best ask remain the same
            volumes[2*i+1,-1] = 0
            for j in range(1, k):
                if orderbook[f'Ask price {j}'].values[i+1] == best_ask + 100*j:
                    volumes[2*i+1, k+j] = orderbook[f'Ask size {j}'].values[i+1]
                else:
                    volumes[2*i+1, k+j] = 0
                if orderbook[f'Bid price {j}'].values[i+1] == best_bid - 100*j:
                    volumes[2*i+1, k-(j+1)] = -orderbook[f'Bid size {j}'].values[i+1]
                else:
                    volumes[2*i+1, k-(j+1)] = 0

        elif orderbook['Ask price 0'].values[i+1] > best_ask:
            # Price increased. Best ask and best bid change
            diff = (orderbook['Ask price 0'].values[i+1] - best_ask) // 100
            volumes[2*i+1,-1] = diff
            if diff > k:
                best_bid = orderbook['Bid price 0'].values[i+1]
                volumes[2*i+1, 2*k-1] = -orderbook['Bid size 0'].values[i+1]
                for l in range(1, 5):
                    if orderbook[f'Bid price {l}'].values[i+1] == best_bid - 100*l:
                        volumes[2*i+1, 2*k-1-l] = -orderbook[f'Bid size {l}'].values[i+1]
                    else:
                        volumes[2*i+1, 2*k-1-l] = 0
            else:
                best_ask = orderbook['Ask price 0'].values[i+1]
                best_bid = orderbook['Bid price 0'].values[i+1]
                volumes[2*i+1, k+diff] = orderbook['Ask size 0'].values[i+1]
                volumes[2*i+1, k-1+diff] = -orderbook['Bid size 0'].values[i+1]
                for j, l in zip(range(1, k-diff), range(1, k)):
                    if orderbook[f'Ask price {l}'].values[i+1] == best_ask + 100*j:
                        volumes[2*i+1, k+diff+j] = orderbook[f'Ask size {l}'].values[i+1]
                    else:
                        volumes[2*i+1, k+diff+j] = 0
                for j, l in zip(range(1, k-1+diff), range(1, k)):
                    if orderbook[f'Bid price {l}'].values[i+1] == best_bid - 100*j:
                        volumes[2*i+1, k-1+diff-j] = -orderbook[f'Bid size {l}'].values[i+1]
                    else:
                        volumes[2*i+1, k-1+diff-j] = 0

        # Price decreased
        elif orderbook['Ask price 0'].values[i+1] < best_ask:
            diff = (best_ask - orderbook['Ask price 0'].values[i+1]) // 100
            volumes[2*i+1, -1] = -diff
            if diff > k:
                best_ask = orderbook['Ask price 0'].values[i+1]
                volumes[2*i+1, 0] = orderbook['Ask size 0'].values[i+1]
                for l in range(1, 5):
                    if orderbook[f'Ask price {l}'].values[i+1] == best_ask + 100*l:
                        volumes[2*i+1, l] = orderbook[f'Ask size {l}'].values[i+1]
                    else:
                        volumes[2*i+1, l] = 0
            else:
                best_ask = orderbook['Ask price 0'].values[i+1]
                best_bid = orderbook['Bid price 0'].values[i+1]
                volumes[2*i+1, k-diff] = orderbook['Ask size 0'].values[i+1]
                volumes[2*i+1, k-diff-1] = -orderbook['Bid size 0'].values[i+1]
                for j,l in zip(range(1, k+diff), range(1, k)):
                    if orderbook[f'Ask price {j}'].values[i+1] == best_ask + 100*j:
                        volumes[2*i+1, k-diff+j] = orderbook[f'Ask size {l}'].values[i+1]
                    else:
                        volumes[2*i+1, k-diff+j] = 0
                for j, l in zip(range(1, k-diff), range(1, k)):
                    if orderbook[f'Bid price {l}'].values[i+1] == best_bid - 100*j:
                        volumes[2*i+1, k-diff-1-j] = -orderbook[f'Bid size {l}'].values[i+1]
                    else:
                        volumes[2*i+1, k-diff-1-j] = 0
        else:
            logging.info('Something wrong in the LOB snapshots function. Check the code logic.')
    return volumes

def compute_spread(orderbook):
    '''This function computes the spread of the orderbook dataframe.'''
    spread = orderbook['Ask price 0'] - orderbook['Bid price 0']
    values, counts = np.unique(spread, return_counts=True)
    for i in range(len(values)):
        logging.info(f'Spread {values[i]} has {counts[i]} occurrences.')
    return spread

def create_LOB_snapshots(stock, date, N, depth, previous_days=False):
    """
    This function creates the LOB snapshots dataframe with the following features:

    - volumes_ask_i: volume of the i-th ask level
    - volumes_bid_i: volume of the i-th bid level
    - spread: spread of the orderbook

    The values are preprocessed using the preprocessing_orderbook_df function. Then, they are normalized 
    through the transformation \( x \rightarrow \text{sign}(x)\sqrt{|x|} \times 0.1 \).

    Parameters
    ----------
    N : int
        Day number of number of days to consider. Check previous_days parameter.
    previous_days : bool, optional
        If True, consider all the data from the 0th to the Nth day. 
        If False, consider only the Nth day. The default is True.

    Returns
    -------
    orderbook_df : pandas.DataFrame
        Dataframe containing the orderbook data with the specified features.
    """

    # Read the dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    orderbook_df_paths = [path for path in dataframes_paths if 'orderbook' in path]
    orderbook_df_paths.sort()
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()

    if previous_days:
        orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][:N]
        message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][:N]
        logging.info(f'Loaded {N} days of data')
        logging.info(f'Original shapes of the dataframes:\n\t{[df.shape for df in orderbook_dfs]}')
        # Preprocess the data using preprocessing_orderbook_df
        orderbook_dfs, _ = zip(*[(preprocessing_orderbook_df(df, msg)) for df, msg in zip(orderbook_dfs, message_dfs)])
        # Merge all the dataframes into a single one
        orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
    else:
        orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][N-1]
        message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][N-1]
        logging.info(f'Loaded {N}th day of data')
        logging.info(f'Original shape of the dataframes:\n\t{orderbook_dfs.shape}')
        orderbook_df, _ = preprocessing_orderbook_df(orderbook_dfs, message_dfs, discard_time=1800)
    
    logging.info(f'Preprocessed shape of the dataframes:\n\t{orderbook_df.shape}')
    # Compute the spread
    spread = compute_spread(orderbook_df)
    # Save the spread
    if not os.path.exists(f'../data/{stock}_{date}/miscellaneous'):
        os.makedirs(f'../data/{stock}_{date}/miscellaneous')
    np.save(f'../data/{stock}_{date}/miscellaneous/spread.npy', spread)

    # Compute the volumes considering also empty levels.
    volumes = LOB_snapshots(orderbook_df, depth, 1, 1)

    # Create the dataframe
    columns = [f'BidVol_{i}' for i in range(depth-1, -1,-1)] + [f'AskVol_{i}' for i in range(0, depth)]
    snapshots_df = pd.DataFrame(volumes[:, :-1], columns=columns)
    price_change = volumes[:,-1]
    np.save(f'../data/{stock}_{date}/miscellaneous/price_change_{N}.npy', price_change)

    # Add the spread
    # snapshots_df['spread'] = spread

    # Normalize the data
    c = 50
    snapshots_df = snapshots_df.applymap(lambda x: math.copysign(1,x)*np.sqrt(np.abs(x))/c)

    # Save the dataframe
    snapshots_df.to_parquet(f'../data/{stock}_{date}/miscellaneous/snapshots_df_{N}.parquet')

    return snapshots_df, price_change

@nb.jit(nopython=True)
def divide_into_windows(data, window_size):
    """Divide the time series into windows of length window_size, each shifted by one time step.
    
    Parameters
    ----------
    data : numpy array or list
        Array containing the data (like training data for instance).
    window_size : int
        Length of the window.
    
    Returns
    -------
    windows : list
        List of the windows."""

    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i+window_size, :])
    return windows

def preprocessing_message_df(message_df, discard_time, sampling_freq):
    '''This function takes as input the message dataframe provided by LOBSTER.com and preprocessed via
    rename_columns. It performs the following operations:
    - Discard the first and last rows of the dataframe, since they are likely "non stationary"
    - Sample the dataframe every f events, where f is the sampling frequency
    - Reset the index
    - Drop the columns Order ID and Time
    
    Parameters
    ----------
    message_df : pandas dataframe
        Message dataframe preprocessed via rename_columns.
    discard_time : int
        Number of seconds to discard from the beginning and the end of the dataframe.
    sampling_freq : int
        Sampling frequency.
    
    Returns
    -------
    message_df : pandas dataframe
        Preprocessed message dataframe.
    message_df.index : pandas index
        Index of the preprocessed message dataframe.
    [k, l] : list
        List of two indexes: one corresponds to the 30th minute of the day and the other to the 30-minute-to-end minute.
    '''
    # Check the Time column and find the index k such that (message['Time'][k] - message['Time'][0])=discard_time
    start = message_df['Time'].values[0]
    end = message_df['Time'].values[-1]

    for i in range(len(message_df)):
        if message_df['Time'].values[i] - start >= discard_time:
            k = i
            break
    
    for i in range(len(message_df)):
        if end - message_df['Time'].values[-i] >= discard_time:
            l = len(message_df) - i
            break

    # Discard the first k and the last l rows
    message_df = message_df.iloc[k:l, :]
    # Sample the orderbook every f events
    message_df = message_df.iloc[::sampling_freq, :]
    # Reset the index
    message_df = message_df.reset_index(drop=True)
    # Drop columns Order ID, Time
    message_df = message_df.drop(columns=['Order ID', 'Time'])
    # Filter by Event type equal to 1,2,3 or 4
    message_df = message_df[message_df['Event type'].isin([1,2,3,4])]

    return message_df, message_df.index, [k, l]

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
    data_length : int
        The length of the data.
    """
    piece_size = int(len(data) / num_pieces + (1 - 1/num_pieces) * overlap_size)
    pieces = []
    for i in range(num_pieces):
        start = i * (piece_size - overlap_size)
        if i < num_pieces - 1:
            end = start + piece_size
        else:
            end = len(data)
        pieces.append(data[start:end])
    
    data_length = 0
    for piece in pieces:
        data_length += len(piece)
    return pieces, data_length

def train_test_split(data, train_size=0.75):
    """Split the data into training and test sets.

    Parameters
    ----------
    data : array_like
        The data to be split.
    train_size : float
        The fraction of the data to be used for training. Default is 0.75.

    Returns
    -------
    train_data : array_like
        The training data.
    test_data : array_like
        The test data.
    """
    train_data = data[:int(np.ceil(train_size * len(data)))]
    test_data = data[int(np.ceil(train_size * len(data))):]
    return train_data, test_data

def compute_accuracy(outputs):
    '''This function computes the accuracy of the discriminator.

    Parameters
    ----------
    outputs : list
        List containing the outputs of the discriminator.
    
    Returns
    -------
    total_accuracy : float
        Total accuracy of the discriminator.'''
    if len(outputs) == 2:
        real_output, fake_output = outputs
        real_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(real_output, 0.5), tf.float32))
        fake_accuracy = tf.reduce_mean(tf.cast(tf.less(fake_output, 0.5), tf.float32))
        total_accuracy = (real_accuracy + fake_accuracy) / 2.0
    else:
        fake_output = outputs
        total_accuracy = tf.reduce_mean(tf.cast(tf.less(fake_output, 0.5), tf.float32))
    return total_accuracy

def simple_rounding(n):
    '''This function takes as input a numpy array and rounds each element to the nearest 100.
    It is used to round the price values to the nearest price level (prices are discretized).
    
    Parameters
    ----------
    n : numpy array
        Array containing the values to be rounded.
    
    Returns
    -------
    new_n : numpy array
        Array containing the rounded values.'''
    new_n = np.zeros_like(n)
    for i,x in enumerate(n):
        r = x%100
        if r < 50:
            new_n[i] = x - r
        else:
            new_n[i] = x + (100-r)
    return new_n

def transform_and_reshape(tensor, T_gen, n_features, c=50):
    '''This function takes as input a tensor of shape (batch_size, T_real, n_features) and transforms it
    into a tensor of shape (batch_size, T_real, n_features) by applying the transformation x -> x^2 * sign(x).
    This is the inverse transformation of the normalization applied to the data.
    
    Parameters
    ----------
    tensor : tensorflow tensor
        Tensor to be transformed.
    T_gen : int
        Number of time steps of the generated samples.
    n_features : int
        Number of features of the generated samples.
    c : int, optional
        Constant equal to the inverse of the normalization constant. The default is 10.
    
    Returns
    -------
    tensor_flat : tensorflow tensor
        Transformed tensor.
    '''
    tensor_flat = tf.reshape(tensor, [-1, T_gen * n_features]).numpy()
    tensor_flat = np.array([[x**2 * math.copysign(1, x) for x in row] for row in tensor_flat])*c
    tensor_flat = tf.reshape(tensor_flat, [-1, T_gen, n_features])
    return tensor_flat

def plot_samples(dataset, generator_model, features, T_gen, n_features_gen, job_id, epoch, args):
    '''This function plots several metrics that track the training process of the GAN. Specifically, it plots:
    - The generated samples
    - The average LOB shape together with the p-values of the Welch's t-test
    - The correlation matrix
    - The histograms of the generated and real samples
    Plots are saved into plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}

    Parameters
    ----------
    dataset : tensorflow dataset
        Dataset containing the real samples. The dataset must be of the form (batch_condition, batch).
    generator_model : tensorflow model
        Generator model.
    noises : list
        List containing the noises used to generate the samples.
    features : list
        List containing the features of the data.
    T_gen : int
        Number of time steps of the generated samples.
    n_features_gen : int
        Number of features of the generated samples.
    job_id : string
        Job id.
    epoch : int
        Epoch number.
    args : argparse object
        Object containing the arguments passed to the script.

    Returns
    -------
    None.

    '''
    # Create two lists to store the generated and real samples. 
    # These list will be of shape (batch_size*number_of_batches_plot, T_gen, n_features_gen)
    generated_samples = []
    real_samples = []

    if args.conditional:
        for batch_condition, batch in dataset:
            noise = tf.random.normal([batch_condition.shape[0], T_gen*args.latent_dim, batch.shape[2]])
            # logging.info(f'LOB at t:\n{batch_condition}\n')
            # logging.info(f'LOB at t+1:\n{batch}\n')
            gen_sample = generator_model([noise, batch_condition])
            # logging.info(f'Generated LOB at t+1:\n{gen_sample}\n')
            # logging.info('-----------------------------------------------------------------')
            if not args.synthetic:
                gen_sample = transform_and_reshape(gen_sample, T_gen, n_features_gen)
                batch = transform_and_reshape(batch, T_gen, n_features_gen)
            for i in range(gen_sample.shape[0]):
                # All the appended samples will be of shape (T_gen, n_features_gen)
                generated_samples.append(gen_sample[i, -1, :])
                real_samples.append(batch[i, -1, :])
    else:
        for batch in dataset:
            noise = tf.random.normal([batch_condition.shape[0], T_gen*args.latent_dim, batch[0].shape[2]])
            # logging.info(f'LOB at t:\n{batch_condition}\n')
            # logging.info(f'LOB at t+1:\n{batch}\n')
            gen_sample = generator_model(noise)
            # logging.info(f'Generated LOB at t+1:\n{gen_sample}\n')
            # logging.info('-----------------------------------------------------------------')
            if not args.synthetic:
                gen_sample = transform_and_reshape(gen_sample, T_gen, n_features_gen)
                batch = transform_and_reshape(batch, T_gen, n_features_gen)
            for i in range(gen_sample.shape[0]):
                # All the appended samples will be of shape (T_gen, n_features_gen)
                generated_samples.append(gen_sample[i, -1, :])
                real_samples.append(batch[i, -1, :])

    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)
    
    means_real, means_gen, p_values = [], [], []
    # Distribution of the generated samples
    fig, axes = plt.subplots(n_features_gen, 1, figsize=(13, 10), tight_layout=True)
    # Time series sample
    fig1, axes1 = plt.subplots(n_features_gen, 1, figsize=(13, 10), tight_layout=True)

    for i, feature in enumerate(features):
        if not args.synthetic:
            d_gen = np.round(generated_samples[:, i].flatten())
        else:
            d_gen = generated_samples[:, i].flatten()
        d_real = real_samples[:, i].flatten()
        _, p_value = ttest_ind(d_gen, d_real, equal_var=False)
        # Compute the means for the average LOB shape
        means_real.append(np.mean(d_real))
        means_gen.append(np.mean(d_gen))
        p_values.append(p_value)
        if n_features_gen == 1:
            axes.plot(d_gen[:500], label='Generated', alpha=0.85)
            axes.plot(d_real[:500], label='Real', alpha=0.85)
            axes.set_title(f'Generated {feature}_{epoch}')
            axes.legend()
            axes1.hist(d_gen, bins=100, label='Generated', alpha=0.75)
            axes1.hist(d_real, bins=100, label='Real', alpha=0.75)
            axes1.set_title(f'Generated {feature}_{epoch}')
            axes1.set_yscale('log')
            axes1.legend()
            axes.set_xlabel('Time (Events)')
            axes1.set_xlabel('Values')
        else:
            axes[i].plot(d_gen[:1000], label='Generated', alpha=0.85)
            axes[i].plot(d_real[:1000], label='Real', alpha=0.85)
            axes[i].set_title(f'Generated {feature}_{epoch}')
            axes[i].legend()
            axes1[i].hist(d_gen, bins=100, label='Generated', alpha=0.75)
            axes1[i].hist(d_real, bins=100, label='Real', alpha=0.75)
            axes1[i].set_title(f'Generated {feature}_{epoch}')
            axes1[i].set_yscale('log')
            axes1[i].legend()
            axes[i].set_xlabel('Time (Events)')
            axes1[i].set_xlabel('Values')
    path = f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}'
    fig.savefig(f'{path}/7_generated_samples_{epoch}.png')
    fig1.savefig(f'{path}/3_generated_samples_hist_{epoch}.png')
    plt.close()

    width = 0.4  # Width of the bars
    indices = np.arange(len(features))  # Create indices for the x position
    # indices1 = np.concatenate((np.arange(1,len(features),2)[::-1], np.arange(0,len(features),2)))

    data = list(zip(features, p_values))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].bar(indices - width/2, np.array(means_gen)[indices], width=width, label='Generated', alpha=0.85)  # Adjust x position for 'Generated' bars
    axes[0].bar(indices + width/2, np.array(means_real)[indices], width=width, label='Real', alpha=0.85)  # Adjust x position for 'Real' bars
    axes[0].set_title(f'Average_LOB_shape_{epoch}')
    axes[0].set_xlabel('Levels')
    axes[0].set_xticks(indices)  # Set the x-tick labels to your features
    axes[0].set_xticklabels(np.array(features)[indices], rotation=60)  # Rotate the x-tick labels by 90 degrees
    axes[0].legend()
    # Remove axes
    axes[1].axis('tight')
    axes[1].axis('off')
    # Create table and add it to the plot
    data = [(feat, f"{p:.2e}") for feat, p in data] # modify data to contain numbers in exponential form with 2 decimal positions
    axes[1].table(cellText=data, cellLoc='center', loc='center')
    # Add a title to the table
    axes[1].set_title(f"Welch's t-test p-values_{epoch}")
    plt.savefig(f'{path}/5_average_LOB_shape_{epoch}.png')
    plt.close()

    correlation_matrix_gen = np.corrcoef(generated_samples, rowvar=False)
    correlation_matrix_real = np.corrcoef(real_samples, rowvar=False)
    if n_features_gen == 1:
        # add a dimension to the correlation matrix
        logging.info(f'Real correlations:\n\t{correlation_matrix_real}')
        logging.info(f'Generated correlations:\n\t{correlation_matrix_gen}')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
        axes[0].imshow(correlation_matrix_gen, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title('Correlation Matrix (generated samples)')
        axes[0].set_xticks(range(generated_samples.shape[1]))
        axes[0].set_yticks(range(generated_samples.shape[1]))
        axes[0].set_xticklabels(features, rotation=90)
        axes[0].set_yticklabels(features)
        # Display the correlation values on the heatmap
        for i in range(correlation_matrix_gen.shape[0]):
            for j in range(correlation_matrix_gen.shape[1]):
                axes[0].text(j, i, round(correlation_matrix_gen[i, j], 2),
                        ha='center', va='center',
                        color='black')
        axes[1].imshow(correlation_matrix_real, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title('Correlation Matrix (real samples)')
        axes[1].set_xticks(range(generated_samples.shape[1]))
        axes[1].set_yticks(range(generated_samples.shape[1]))
        axes[1].set_xticklabels(features, rotation=90)
        axes[1].set_yticklabels(features)
        for i in range(correlation_matrix_real.shape[0]):
            for j in range(correlation_matrix_real.shape[1]):
                axes[1].text(j, i, round(correlation_matrix_real[i, j], 2),
                        ha='center', va='center',
                        color='black')
        plt.savefig(f'{path}/6_correlation_matrix_{epoch}.png')
        plt.close()
    
    return None

def bar_plot_levels(stock, date, N, window_size, c=50):
    condition_train = np.load(f'../data/{stock}_{date}/miscellaneous/condition_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
    input_train = np.load(f'../data/{stock}_{date}/miscellaneous/input_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
    levels = condition_train.shape[2]//2
    values_bid, counts_bid = [], []
    values_ask, counts_ask = [], []
    for i in range(levels):
        bid = np.array([x**2 * math.copysign(1, x)*c for x in condition_train[:,:, i]]).flatten()
        ask = np.array([x**2 * math.copysign(1, x)*c for x in condition_train[:,:, i+levels]]).flatten()
        vb, cb = np.unique(bid, return_counts=True)
        if np.all(vb > 0):
            logging.info(f'✗ Some bid volumes at level {i} are positive at time t!')
        values_bid.append(vb)
        counts_bid.append(cb)
        va, ca = np.unique(ask, return_counts=True)
        values_ask.append(va)
        counts_ask.append(ca)

    fig, axes = plt.subplots(2, levels, figsize=(15, 8), tight_layout=True)
    for i in range(levels):
        axes[0,i].bar(values_bid[i], counts_bid[i], width=10, color='green', alpha=0.7)
        axes[0,i].set_title(f'Bid Volume {levels-i}')
        axes[1,i].bar(values_ask[-i], counts_ask[-i], width=10, color='red', alpha=0.7)
        axes[1,i].set_title(f'Ask Volume {levels-i}')
    fig.suptitle(r'$s_t$')
    fig.savefig(f'plots/bar_plot_t_{stock}_{N}days.png')
    
    values_bid, counts_bid = [], []
    values_ask, counts_ask = [], []
    for i in range(levels):
        bid = np.array([x**2 * math.copysign(1, x)*c for x in input_train[:,:, i]]).flatten()
        ask = np.array([x**2 * math.copysign(1, x)*c for x in input_train[:,:, i+levels]]).flatten()
        vb, cb = np.unique(bid, return_counts=True)
        values_bid.append(vb)
        counts_bid.append(cb)
        va, ca = np.unique(ask, return_counts=True)
        values_ask.append(va)
        counts_ask.append(ca)
    fig1, axes1 = plt.subplots(2, levels, figsize=(15, 8), tight_layout=True)
    for i in range(levels):
        axes1[0,i].bar(values_bid[i], counts_bid[i], width=10, color='green', alpha=0.7)
        axes1[0,i].set_title(f'Level {levels-i}')
        axes1[1,i].bar(values_ask[-i], counts_ask[-i], width=10, color='red', alpha=0.7)
        axes1[1,i].set_title(f'Level {levels-i}')
    fig1.suptitle(r'$s_{t+1}$')
    fig1.savefig(f'plots/bar_plot_tp1_{stock}_{N}days.png')
    return None

def correlation_matrix(dataset, generator_model, noise, T_gen, n_features_gen, job_id, bootstrap_iterations=10000):
    '''This function computes the correlation matrix of the generated samples and the real samples, together with the standard errors
    evaluated using the bootstrap method. The correlation matrix is saved in the plots folder corresponding to job_id.

    Parameters
    ----------
    dataset : tensorflow dataset
        Dataset containing the real samples. The dataset must be of the form (batch_condition, batch).
    generator_model : tensorflow model
        Generator model.
    noises : list
        List containing the noises used to generate the samples.
    best_epoch : int
        Best epoch number.
    scaler : sklearn scaler
        Scaler used to scale the data. It can be None.
    T_gen : int
        Number of time steps of the generated samples.
    n_features_gen : int
        Number of features of the generated samples.
    job_id : string
        Job id.
    bootstrap_iterations : int, optional
        Number of bootstrap iterations. The default is 10000.
    
    Returns
    -------
    None.'''
    generated_samples = []
    real_samples = []
    k = 0
    for batch_condition, batch in dataset:
        batch_size = batch.shape[0]
        noise = tf.random.normal([batch_size, 300, 6])
        gen_sample = generator_model([noise, batch_condition])
        gen_sample = transform_and_reshape(gen_sample, T_gen, n_features_gen)
        batch = transform_and_reshape(batch, T_gen, n_features_gen)
        for i in range(gen_sample.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples.append(gen_sample[i, -1, :])
            real_samples.append(batch[i, -1, :])
        k += 1
    
    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)
    generated_samples = generated_samples.reshape(generated_samples.shape[0], generated_samples.shape[1])
    real_samples = real_samples.reshape(real_samples.shape[0], real_samples.shape[1])

    generated_samples_bootstrap = np.copy(generated_samples)
    real_samples_bootstrap = np.copy(real_samples)

    correlations_gen = np.zeros(shape=(bootstrap_iterations, generated_samples.shape[1], generated_samples.shape[1]))
    correlations_real = np.zeros(shape=(bootstrap_iterations, generated_samples.shape[1], generated_samples.shape[1]))
    for i in range(bootstrap_iterations):
        # Shuffle randomly the generated samples and the real samples
        for i in range(generated_samples_bootstrap.shape[1]):
            np.random.shuffle(generated_samples_bootstrap[:, i])
            np.random.shuffle(real_samples_bootstrap[:, i])
        # Compute the correlation matrix
        correlation_matrix_gen = np.corrcoef(generated_samples_bootstrap, rowvar=False)
        correlation_matrix_real = np.corrcoef(real_samples_bootstrap, rowvar=False)
        correlations_gen[i, :, :] = correlation_matrix_gen
        correlations_real[i, :, :] = correlation_matrix_real

    # Initialize an array to store the stacked elements
    stacked_elements_gen = np.zeros((bootstrap_iterations, n_features_gen*(n_features_gen-1)//2))
    stacked_elements_real = np.zeros((bootstrap_iterations, n_features_gen*(n_features_gen-1)//2))
    # Fill the stacked elements array
    k = 0
    for i in range(n_features_gen):
        for j in range(i+1, n_features_gen):
            # Extract the element at position (i, j) from each matrix and stack them
            stacked_elements_gen[:, k] = [mat[i, j] for mat in correlations_gen]
            stacked_elements_real[:, k] = [mat[i, j] for mat in correlations_real]
            k += 1

    # Now compute the standard deviations for each set of elements
    standard_deviations_gen = np.std(stacked_elements_gen, axis=0)
    standard_deviations_real = np.std(stacked_elements_real, axis=0)
    # Now compute the mean of the elements
    mean_gen = np.mean(stacked_elements_gen, axis=0)
    mean_real = np.mean(stacked_elements_real, axis=0)
    # Create a matrix n_features_gen x n_features_gen that has 0 non the diagonal and the standard deviations on the upper triangle
    std_matrix_gen = np.zeros((n_features_gen, n_features_gen))
    # do the same for the mean
    mean_matrix_gen = np.zeros((n_features_gen, n_features_gen))
    tri_indices = np.triu_indices(n_features_gen, k=1)
    n_elements = int(n_features_gen*(n_features_gen-1)/2)
    std_matrix_gen[tri_indices[0][:n_elements], tri_indices[1][:n_elements]] = standard_deviations_gen
    std_matrix_gen = std_matrix_gen + std_matrix_gen.T
    std_matrix_real = np.zeros((n_features_gen, n_features_gen))
    std_matrix_real[tri_indices[0][:n_elements], tri_indices[1][:n_elements]] = standard_deviations_real
    std_matrix_real = std_matrix_real + std_matrix_real.T
    mean_matrix_gen[tri_indices[0][:n_elements], tri_indices[1][:n_elements]] = mean_gen
    mean_matrix_gen = mean_matrix_gen + mean_matrix_gen.T
    mean_matrix_real = np.zeros((n_features_gen, n_features_gen))
    mean_matrix_real[tri_indices[0][:n_elements], tri_indices[1][:n_elements]] = mean_real
    mean_matrix_real = mean_matrix_real + mean_matrix_real.T


    logging.info(f'Std matrix of the generated samples:\n{std_matrix_gen}')
    logging.info(f'Std matrix of the real samples:\n{std_matrix_real}')

    correlation_matrix_gen = np.corrcoef(generated_samples, rowvar=False)
    correlation_matrix_real = np.corrcoef(real_samples, rowvar=False)
    features = np.array([[f'ask_volume_{i}',f'bid_volume_{i}'] for i in range(1, int(0.5*generated_samples.shape[1])+1)]).flatten()
    _, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].imshow(correlation_matrix_gen, cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title('Correlation Matrix (generated samples)')
    axes[0].set_xticks(range(generated_samples.shape[1]))
    axes[0].set_yticks(range(generated_samples.shape[1]))
    axes[0].set_xticklabels(features, rotation=90)
    axes[0].set_yticklabels(features)
    # Display the correlation values on the heatmap
    k = 0
    for i in range(correlation_matrix_gen.shape[0]):
        for j in range(correlation_matrix_gen.shape[1]):
            text_str = f"{format(correlation_matrix_gen[i, j], '.2f')} \n ({format(mean_matrix_gen[i, j], '.2e')} \n ± {format(2*std_matrix_gen[i, j], '.2e')})"
            axes[0].text(j, i, text_str,
                ha='center', va='center',
                color='black', fontsize=7)
            k += 1
    axes[1].imshow(correlation_matrix_real, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Correlation Matrix (real samples)')
    axes[1].set_xticks(range(generated_samples.shape[1]))
    axes[1].set_yticks(range(generated_samples.shape[1]))
    axes[1].set_xticklabels(features, rotation=90)
    axes[1].set_yticklabels(features)
    for i in range(correlation_matrix_real.shape[0]):
        for j in range(correlation_matrix_real.shape[1]):
            text_str = f"{format(correlation_matrix_real[i, j], '.2f')} \n ({format(mean_matrix_real[i, j], '.2e')} \n ± {format(2*std_matrix_real[i, j], '.2e')})"
            axes[1].text(j, i, text_str,
                    ha='center', va='center',
                    color='black', fontsize=7)
    path = [s for s in os.listdir('plots/') if f'{job_id}' in s][0]
    plt.savefig(f'plots/{path}/6_correlation_matrix_final.png')
    plt.close()
    return None

def plot_pca_with_marginals(dataset_real, dataset_gen, job_id, args):
    # Perform PCA on both datasets
    pca1 = PCA(n_components=2)
    pca2 = PCA(n_components=2)
    principalComponents_real = pca1.fit_transform(dataset_real)
    principalComponents_gen = pca2.fit_transform(dataset_gen)

    # Combine the PCA components into a DataFrame
    df_real = pd.DataFrame(data=principalComponents_real, columns=['PC1', 'PC2'])
    df_real['Type'] = 'Real'
    
    df_gen = pd.DataFrame(data=principalComponents_gen, columns=['PC1', 'PC2'])
    df_gen['Type'] = 'Generated'
    
    # Concatenate the two DataFrames
    df = pd.concat([df_real, df_gen])

    # Use Seaborn's jointplot to plot the scatterplot with marginal histograms
    g = sns.jointplot(data=df, x='PC1', y='PC2', hue='Type', height=10, alpha=0.5) # set alpha to 0.5 for 50% transparency

    # Add a main title
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("First two PCA Components of real and fake dataset with marginal distributions")

    # Save the plot
    plt.savefig(f'plots/{job_id}.pbs01_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/8_pca_with_marginals_day{args.N_days}.png')
    plt.close()

def create_animated_gif(job_id):
    '''This function takes as input the job id and creates an animated gif from the images
    stored in the plots folder.'''
    # Get the folder name
    image_folder = [f for f in os.listdir('plots/') if f'{job_id}' in f]
    if len(image_folder) > 1:
        raise ValueError(f'There are more than one {job_id} folders in the plots folder')
    image_folder = image_folder[0]
    # Choose the output gif path
    output_gif_path = f'plots/{image_folder}/002_{job_id}_0.gif'
    c = 0
    while os.path.exists(output_gif_path):
        c += 1
        output_gif_path = f'plots/{image_folder}/002_{job_id}_{c}.gif'
    
    # Get all files from the folder
    image_files = [f for f in os.listdir(f'plots/{image_folder}') if 'hist' in f]
    logging.info(f'Creating animated GIF from {len(image_files)} images')
    
    image_files.sort()
    logging.info(f'{image_files}')
    images = []

    # Open, append new images to the list, and close the file
    for image_file in image_files:
        image_path = os.path.join(f'plots/{image_folder}', image_file)
        with Image.open(image_path) as img:
            images.append(img.copy())

    # Save the images as an animated GIF
    images[0].save(output_gif_path,
                    save_all=True, append_images=images[1:], optimize=False, duration=int(10000/len(images)), loop=0)

    # Close all the images in the list
    for img in images:
        img.close()

    # Eliminate all the images
    for image_file in image_files:
        os.remove(os.path.join(f'plots/{image_folder}', image_file))

def plot_volumes(orderbook_df, depth, stock, date):
    '''This function create a bar plot displaying the volumes for a random index (random LOB snapshot). 
    The space between the last bid and the first ask must be equal to the spread for that index'''
    depth = depth*2 + 1
    for _ in range(10):
        idx = np.random.randint(0, orderbook_df.shape[0])
        ask_volumes = orderbook_df.iloc[idx, 0:depth:2].values
        bid_volumes = orderbook_df.iloc[idx, 1:depth:2].values
        spread = orderbook_df.iloc[idx, -1]
        s = 0.5

        # Creating positions for bid_volumes and ask_volumes bars on the x-axis
        bid_volumes_positions = np.arange(len(bid_volumes))
        ask_volumes_positions = np.arange(len(ask_volumes)) + len(bid_volumes) + spread  # Offset by the length of bid_volumes and the spacing spread

        bid_volumes_labels = np.arange(-len(bid_volumes), 0)  # Labels for bid_volumes from -len(bid_volumes) to -1
        ask_volumes_labels = np.arange(1, len(ask_volumes) + 1)  # Labels for ask_volumes from 1 to len(ask_volumes)

        # New positions for ask_volumes bars to align with the new labels
        ask_volumes_positions = bid_volumes_labels[-1] + s + 1 + np.arange(len(ask_volumes))

        # Combined labels and positions for the bars
        all_positions = np.concatenate((bid_volumes_labels, ask_volumes_positions))
        all_labels = np.concatenate((bid_volumes_labels, ask_volumes_labels))
        all_values = np.concatenate((bid_volumes, ask_volumes))

        # Plotting with the new labels and positions
        plt.figure()
        plt.bar(all_positions, all_values, color=['blue' if val < 0 else 'orange' for val in all_values])
        plt.xticks(all_positions, all_labels)  # Set custom ticks
        plt.title('LOB snapshot')
        plt.xlabel('Levels')
        plt.ylabel('Volumes')

        # Create dummy bars for the legend
        blue_bar = plt.bar([0], [0], color='blue', label='Bid')
        orange_bar = plt.bar([0], [0], color='orange', label='Ask')

        # Add legend
        plt.legend()

        plt.savefig(f'plots/{stock}_{date}_LOB_snapshot_{idx}.png')
        plt.close()

def ar1_fit(data):
    # Fit an AR(1) model to the data
    model = AutoReg(data, lags=1)
    model_fit = model.fit()
    residuals = model_fit.resid
    phi = model_fit.params[1]

    # Perform normaltest on the residuals
    logging.info("D'AGOSTINO-PEARSON TEST")
    logging.info('Null hypothesis: x comes from a normal distribution')
    _, p = normaltest(residuals)
    logging.info(f'p = {p}')
    if p < 0.05:  # null hypothesis: x comes from a normal distribution
        logging.info(f"\tThe null hypothesis can be rejected")
    else:
        logging.info(f"\tThe null hypothesis cannot be rejected")
    
    # Perform Ljung-Box test on the residuals
    logging.info("LJUNG-BOX TEST")
    logging.info('Null hypothesis: there is no significant autocorellation')
    _, p = acorr_ljungbox(residuals, lags=1)
    logging.info(f'p = {p}')
    if p < 0.05:  # null hypothesis: there is no significant autocorellation
        logging.info(f"\tThe null hypothesis can be rejected")
    else:
        logging.info(f"\tThe null hypothesis cannot be rejected")
    return None

def anomaly_injection_orderbook(fp, T_condition, depth, ampl, anomaly_type):
    '''This function takes as input a orderbook dataframe and injects anomalies in it. There are
    two types of anomalies: the first one is the random submissions of enormous orders (buy or sell)
    at a random level, while the second one consists in filling one side of the LOB with a lot of orders.
    The anomalies are injected in the 0.5% of the windows. An anomaly is set as the mean of the level
    plus/minus the standard deviation of the level multiplied by a constant ampl.
    
    Parameters
    ----------
    fp : numpy array
        Array containing the data.
    means : numpy array
        Array containing the means of each level.
    stds : numpy array
        Array containing the standard deviations of each level.
    T_condition : int
        Number of time steps of the condition.
    depth : int
        Number of levels of the LOB.
    ampl : float
        Constant used to amplify the standard deviation.
    anomaly_type : string
        Type of anomaly to inject. It can be 'big_order', 'fill_side', 'liquidity_crisis'.
    
    Returns
    -------
    data : numpy array
        Array containing the data with the anomalies injected.
    chosen_windows : list
        List containing the indexes of the windows where the anomalies have been injected.'''
    # create a copy of fp
    data = fp.copy()
    # Select the % of the data to inject anomalies
    n_anomalies = int(len(data) * 0.005)
    chosen_windows = []
    for _ in range(n_anomalies):
        chosen_window = np.random.randint(0, data.shape[0])
        chosen_windows.append(chosen_window)

        if anomaly_type == 'big_order':
            chosen_timestamp = np.random.randint(T_condition, data.shape[1], 1)
            chosen_feature = np.random.randint(0, (2*depth))
            # evaluate the interquartile range of the chosen feature
            # third_quantile = np.quantile(data[:, :, chosen_feature].flatten(), 0.75)
            # first_quantile = np.quantile(data[:, :, chosen_feature].flatten(), 0.25)
            # iqr = np.abs(third_quantile - first_quantile)
            if chosen_feature % 2 == 0:
                anomaly = np.sqrt(1000*ampl)/10
            else:
                anomaly = -np.sqrt(1000*ampl)/10
            data[chosen_window, chosen_timestamp, chosen_feature] = anomaly

        if anomaly_type == 'fill_side':
            chosen_timestamp = np.random.choice(np.arange(T_condition, data.shape[1]), 10, replace=False)
            random_number_1 = np.random.normal()
            chosen_feature = None
            if random_number_1 <= 0:
                data[chosen_window, chosen_timestamp, 1:2*depth+1:2] = -np.sqrt(1000*ampl)/10
            else:
                data[chosen_window, chosen_timestamp, 1:2*depth:2] = np.sqrt(1000*ampl)/10
        
        if anomaly_type == 'liquidity_crisis':
            chosen_timestamp = np.random.choice(np.arange(T_condition, data.shape[1]), 10, replace=False)
            random_number_1 = np.random.normal()
            chosen_feature = None
            if random_number_1 <= 0:
                data[chosen_window, chosen_timestamp, 1:2*depth+1:2] = 0
            else:
                data[chosen_window, chosen_timestamp, 1:2*depth:2] = 0
    return data, chosen_windows, chosen_feature

def plot_weightwatcher(details, epoch, job_id):
    # Collect alpha values
    alpha_values = [details['alpha'][i] for i in range(len(details['alpha']))]

    # Plot the distribution of alpha values
    plt.figure(figsize=(10, 6))
    plt.hist(alpha_values, bins=50, alpha=0.7, color='blue')
    plt.title('Distribution of Alpha Values')
    plt.xlabel('Alpha')
    plt.ylabel('Frequency')

    # Specify the path and filename to save the figure
    path = [s for s in os.listdir('plots/') if f'{job_id}' in s][0] 

    # Save the plot to the specified file
    plt.savefig(f'plots/{path}/7_alpha_distribution_{epoch}.png')

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script several functions used to pre and post process the data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    
    if os.getenv("PBS_JOBID") != None:
        job_id = os.getenv("PBS_JOBID")
    else:
        job_id = os.getpid()
    
    logging.basicConfig(filename=f'logs/data_utils_{job_id}.log', format='%(message)s', level=levels[args.log])

    logger = tf.get_logger()
    logger.setLevel('ERROR')

    np.random.seed(666)

    # Print the current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Load the data
    stock = 'MSFT'
    date = '2018-04-01_2018-04-30_5'

    bar_plot_levels(stock, date, c=50)
    exit()
    
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    orderbook_df_paths = [path for path in dataframes_paths if 'orderbook' in path]
    orderbook_df_paths.sort()
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()

    orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][0]
    message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][0]
    for sec in np.arange(0,10,0.5):
        # Preprocess the data using preprocessing_orderbook_df
        dfs, _ = zip(*[(preprocessing_orderbook_df(df, msg, sampling_seconds=sec, discard_time=1800)) for df, msg in zip(orderbook_dfs, message_dfs)])
        # Merge all the dataframes into a single one
        orderbook_df = pd.concat(dfs, ignore_index=True)
        best_ask = orderbook_df['Ask price 0'].values//100
        ret = np.diff(best_ask)
        values, counts = np.unique(ret, return_counts=True)
        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.bar(values, counts)
        plt.savefig(f'plots/ret_{sec}.png')
        plt.close()
    

