'''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
composed by a sliding window that shifts by one time step each time.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import logging
import numba as nb
import tensorflow as tf
from PIL import ImageDraw
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest, ttest_ind
from statsmodels.tsa.stattools import adfuller
import math
from sklearn.decomposition import PCA
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

def rename_columns(dataframes_folder_path):
    '''This function takes as input the path of the folder containing the dataframes and renames the columns
    of the message and orderbook dataframes. Then, it saves the dataframes in the same folder.
    
    Parameters
    ----------
    dataframes_folder_path : string
        Path of the folder containing the dataframes.
    
    Returns
    -------
    None.'''

    # Read the message dataframes
    dataframes_paths = os.listdir(f'{dataframes_folder_path}/')
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


def preprocessing_orderbook_df(orderbook, message, discard_time):
    '''This function takes as input the orderbook dataframe provided by LOBSTER.com and preprocessed via
    rename_columns. It performs the following operations:
    - Discard the first and last rows of the dataframe, since they are likely "non stationary"
    - Sample the dataframe every 10 seconds
    - Reset the index
    - Drop the columns Order ID and Time
    
    Parameters
    ----------
    orderbook : pandas dataframe
        Orderbook dataframe preprocessed via rename_columns.
    message : pandas dataframe
        Message dataframe preprocessed via rename_columns.
    discard_time : int
        Number of seconds to discard from the beginning and the end of the dataframe.
    
    Returns
    -------
    orderbook : pandas dataframe
        Preprocessed orderbook dataframe.
    [k, l] : list
        List of two indexes: one corresponds to the 30th minute of the day and the other to the 30-minute-to-end minute.'''
    # Take the Time column of message and add it to orderbook
    orderbook['Time'] = message['Time']
    # Check the Time column and find the index k such that (orderbook['Time'][k] - orderbook['Time'][0])=discard_time
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
    # Iterate through the dataframe to select rows approximately 10 seconds apart
    for i in range(1, len(orderbook)):
        if orderbook['Time'].values[i] - current_time >= 10:
            selected_indices.append(i)
            current_time = orderbook['Time'].values[i]
    # Create a new dataframe with the selected rows
    orderbook = orderbook.iloc[selected_indices]
    orderbook = orderbook.reset_index(drop=True)
    return orderbook, [k, l]

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
    for row in range(ask_prices.shape[0]):
        start_ask = ask_prices[row, 0] # best ask price
        start_bid = bid_prices[row, 0] # best bid price
        volume_ask[row, 0] = ask_volumes[row, 0] # volume at best ask
        volume_bid[row, 0] = bid_volumes[row, 0] # volume at best bid
        for i in range(1,depth): #from the 2th until the 4th tick price level
            if ask_prices[row, i] == start_ask + 100*i: # check if the next occupied level is one tick ahead
                volume_ask[row, i] = ask_volumes[row, i]
            else:
                volume_ask[row, i] = 0

            if bid_prices[row, i] == start_bid - 100*i: # check if the next occupied level is one tick before
                volume_bid[row, i] = bid_volumes[row, i]
            else:
                volume_bid[row, i] = 0
    return volume_ask, -volume_bid


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

def transform_and_reshape(tensor, T_gen, n_features, c=10):
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


def plot_samples(dataset, generator_model, noise, features, T_gen, n_features_gen, job_id, epoch, args):
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

    k = 0
    for batch_condition, batch in dataset:
        gen_sample = generator_model([noise[k], batch_condition])
        gen_sample = transform_and_reshape(gen_sample, T_gen, n_features_gen)
        batch = transform_and_reshape(batch, T_gen, n_features_gen)
        for i in range(gen_sample.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples.append(gen_sample[i, -1, :])
            real_samples.append(batch[i, -1, :])
        k += 1

    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)
    
    means_real, means_gen, p_values = [], [], []
    fig, axes = plt.subplots(n_features_gen, 1, figsize=(10, 10), tight_layout=True)
    fig1, axes1 = plt.subplots(n_features_gen, 1, figsize=(10, 10), tight_layout=True)
    for i, feature in enumerate(features):
        d_gen = np.round(generated_samples[:, i].flatten())
        d_real = real_samples[:, i].flatten()
        _, p_value = ttest_ind(d_gen, d_real, equal_var=False)
        means_real.append(np.mean(d_real))
        means_gen.append(np.mean(d_gen))
        p_values.append(p_value)
        axes[i].plot(d_gen[:200], label='Generated', alpha=0.85)
        axes[i].plot(d_real[:200], label='Real', alpha=0.85)
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
    indices1 = np.concatenate((np.arange(1,len(features),2)[::-1], np.arange(0,len(features),2)))
    print(np.array(features)[indices1])

    data = list(zip(features, p_values))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].bar(indices - width/2, np.array(means_gen)[indices1], width=width, label='Generated', alpha=0.85)  # Adjust x position for 'Generated' bars
    axes[0].bar(indices + width/2, np.array(means_real)[indices1], width=width, label='Real', alpha=0.85)  # Adjust x position for 'Real' bars
    axes[0].set_title(f'Average_LOB_shape_{epoch}')
    axes[0].set_xlabel('Levels')
    axes[0].set_xticks(indices)  # Set the x-tick labels to your features
    axes[0].set_xticklabels(np.array(features)[indices1], rotation=60)  # Rotate the x-tick labels by 90 degrees
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

    # generated_samples = generated_samples.reshape(generated_samples.shape[0]*generated_samples.shape[1], generated_samples.shape[2])
    # real_samples = real_samples.reshape(real_samples.shape[0]*real_samples.shape[1], real_samples.shape[2])
    correlation_matrix_gen = np.corrcoef(generated_samples, rowvar=False)
    correlation_matrix_real = np.corrcoef(real_samples, rowvar=False)
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
        # noise = tf.random.normal([batch_size, 300, 6])
        gen_sample = generator_model([noise[k], batch_condition])
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
    # Create a matrix n_features_gen x n_features_gen that has 0 non the diagonal and the standard deviations on the upper triangle
    std_matrix_gen = np.zeros((n_features_gen, n_features_gen))
    tri_indices = np.triu_indices(n_features_gen, k=1)
    n_elements = int(n_features_gen*(n_features_gen-1)/2)
    std_matrix_gen[tri_indices[0][:n_elements], tri_indices[1][:n_elements]] = standard_deviations_gen
    std_matrix_gen = std_matrix_gen + std_matrix_gen.T
    std_matrix_real = np.zeros((n_features_gen, n_features_gen))
    std_matrix_real[tri_indices[0][:n_elements], tri_indices[1][:n_elements]] = standard_deviations_real
    std_matrix_real = std_matrix_real + std_matrix_real.T

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
            text_str = f"{format(correlation_matrix_gen[i, j], '.2f')} \n ({format(2*std_matrix_gen[i, j], '.2e')})"
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
            text_str = f"{format(correlation_matrix_real[i, j], '.2f')} \n ({format(2*std_matrix_real[i, j], '.2e')})"
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


def sin_wave(amplitude, omega, phi, change_amplitude=False):
    '''This function generates a sine wave with the specified parameters.
    
    Parameters
    ----------
    amplitude : float
        Amplitude of the sine wave.
    omega : float
        Frequency of the sine wave.
    phi : float
        Phase of the sine wave.
    change_amplitude : bool, optional
        If True, the amplitude of the sine wave will be modified every 3 periods. The default is False.
    
    Returns
    -------
    sine_wave : numpy array
        Sine wave with the specified parameters.'''
    # Parameters
    num_periods = 1000  # Total number of periods to generate
    samples_per_period = 100  # Number of samples per period

    # Generate a time vector
    time = np.linspace(0, 2 * np.pi * num_periods, samples_per_period * num_periods)

    # Generate the sine wave
    sine_wave = amplitude * np.sin(omega*time + phi)

    # Modify the amplitude every 3 periods
    if change_amplitude:
        for i in range(num_periods):
            if i % 5 == 2:  # Check if it is the third period (0-based index)
                sine_wave[i * samples_per_period:(i + 1) * samples_per_period] *= 3
    # sine_wave = np.reshape(sine_wave, (sine_wave.shape[0], 1))
    return sine_wave

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

def step_fun(freq):
    # Parameters
    num_periods = 1000  # Total number of periods to generate
    samples_per_period = 100  # Number of samples per period

    # Generate a time vector
    time = np.linspace(0, 2 * np.pi * num_periods, samples_per_period * num_periods)

    # Generate the step function
    step = np.zeros(time.shape)
    for t in range(len(time)):
        if time[t] % (np.pi) < np.pi/(freq*2):
            step[t] = np.random.randint(0, 5)
    return step

def ar1():
    # Parameters
    num_periods = 1000  # Total number of periods to generate
    samples_per_period = 100  # Number of samples per period
    phi = 0.9  # AR(1) coefficient
    mu = 0.4
    time = np.linspace(0, 2 * np.pi * num_periods, samples_per_period * num_periods)

    # Generate the AR(1) process
    ar1 = np.zeros(time.shape)
    for t in range(1, len(time)):
        ar1[t] = mu + phi * ar1[t - 1] + np.random.normal(0, 0.5)
    return ar1

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

def compute_spread(orderbook):
    '''This function computes the spread of the orderbook dataframe.'''
    spread = orderbook['Ask price 1'] - orderbook['Bid price 1']
    return spread

def create_orderbook_dataframe(N, previos_days=True):
    '''This function create the orderbook dataframe with the following features:
    - volumes_ask_i: volume of the i-th ask level
    - volumes_bid_i: volume of the i-th bid level
    - spread: spread of the orderbook
    
    The values are preprocessed using the preprocessing_orderbook_df function. Then, they are normalized
    through the transformation x -> sign(x)*sqrt(|x|)*0.1.
    Parameters
    ----------
    N : int
        Number of days to consider.
    previos_days : bool, optional
        If True, the data of the previous days are considered. The default is True.
    
    Returns
    -------
    orderbook_df : pandas dataframe
        Dataframe containing the orderbook data with the specified features.'''
    # Load data
    stock = 'MSFT'
    date = '2018-04-01_2018-04-30_5'
    total_depth = 5

    # Read the dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    orderbook_df_paths = [path for path in dataframes_paths if 'orderbook' in path]
    orderbook_df_paths.sort()
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()
    if previos_days:
        orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][:N]
        message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][:N]
        # Preprocess the data using preprocessing_orderbook_df
        orderbook_dfs, _ = zip(*[(preprocessing_orderbook_df(df, msg, discard_time=1800)) for df, msg in zip(orderbook_dfs, message_dfs)])
        # Merge all the dataframes into a single one
        orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
    else:
        orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][N]
        message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][N]
        orderbook_df, _ = preprocessing_orderbook_df(orderbook_dfs, message_dfs, discard_time=1800)

    
    # Compute the spread
    spread = compute_spread(orderbook_df)
    # Extract the prices and volumes
    bid_prices, bid_volumes, ask_prices, ask_volumes = prices_and_volumes(orderbook_df)
    # Compute the volumes considering also empty levels
    volumes_ask, volumes_bid = volumes_per_level(ask_prices, bid_prices, ask_volumes, bid_volumes, total_depth)
    # Create a dataframe with the volumes
    orderbook_df = pd.DataFrame()
    for i in range(total_depth):
        orderbook_df[f'volumes_ask_{i+1}'] = volumes_ask[:, i]
        orderbook_df[f'volumes_bid_{i+1}'] = volumes_bid[:, i]
    # Add the spread
    orderbook_df['spread'] = spread
    # Normalize the data
    orderbook_df = orderbook_df.applymap(lambda x: math.copysign(1,x)*np.sqrt(np.abs(x))*0.1)
    return orderbook_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script several functions used to pre and post process the data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-N', '--N_days', type=int, help='Number of the day to consider')
    parser.add_argument('-d', '--depth', help='Depth of the orderbook', type=int)
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int)
    parser.add_argument('-ld', '--latent_dim', help='Latent dimension', type=int)
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of generator layers', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of discriminator layers', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-sc', '--skip_connection', action='store_true', help='Use or not skip connections')
    parser.add_argument('-Tc', '--T_condition', help='Number of time steps to condition on', type=int, default=2)
    parser.add_argument('-Tg', '--T_gen', help='Number of time steps to generate', type=int, default=1)
    parser.add_argument('-ls', '--loss', help='Loss function (original, wasserstein)', type=str, default='original')
    parser.add_argument('-lo', '--load', help='Load a model. The job_id must be provided', type=int)

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
    
    logging.basicConfig(filename=f'data_utils_{job_id}.log', format='%(message)s', level=levels[args.log])

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
    batch_size = args.batch_size
    T_gen = args.T_gen
    T_condition = args.T_condition
    latent_dim = args.latent_dim*T_gen
    n_features_gen = 6
    window_size = T_gen + T_condition
    N = args.N_days
    prev_job_id = args.load

    logging.info(f'Loaded model:\n\t{prev_job_id}')
    
    # Check if the orderflow is stationary or not
    # N = 10
    # p_values = np.zeros((N,6))
    # for n in range(N):
    #     logging.info(f'Day {n}')
    #     orderbook = create_orderbook_dataframe(N)
    #     columns = orderbook.columns
    #     for col in range(6):
    #         res = adfuller(orderbook[columns[col]].values)
    #         p_values[n, col] = res[1]
    #         logging.info(f'p-value for {columns[col]}: {res[1]}')
    
    # # Visualize p_values matrix using imshow with a colorbar
    # fig, ax = plt.subplots(figsize=(10, 10))
    # im = ax.imshow(p_values, cmap='viridis_r', vmin=0, vmax=1)
    # ax.set_title('ADF test p-values')
    # ax.set_xlabel('Features')
    # ax.set_ylabel('Days')
    # ax.set_xticks(np.arange(len(columns[:6])))
    # ax.set_yticks(np.arange(N))
    # ax.set_xticklabels(columns[:6])
    # ax.set_yticklabels(np.arange(N))
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    # # Loop over data dimensions and create text annotations.
    # for i in range(N):
    #     for j in range(6):
    #         text = ax.text(j, i, f"{p_values[i, j]:.2e}",
    #                     ha="center", va="center", color="black", fontsize=11)
    # fig.tight_layout()
    # plt.savefig(f'plots/ADF_test_p_values.png')


    logging.info('Loading input_train, input_validation and input_test sets...')
    input_train = np.load(f'../data/input_train_{stock}_{window_size}_day{N}_orderbook.npy', mmap_mode='r')
    condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_day{N}_orderbook.npy', mmap_mode='r')

    logging.info(f'input_train shape:\n\t{input_train.shape}')
    logging.info(f'condition_train shape:\n\t{condition_train.shape}')
    dataset_train = tf.data.Dataset.from_tensor_slices((condition_train, input_train)).batch(batch_size)
    
    # Load the best generator
    generator_model = tf.keras.models.load_model(f'models/{prev_job_id}.pbs01_dense_dense_3_3_50_original/generator_model.h5')
    
    # generator_model.summary(print_fn=logging.info)
    
    generated_samples = []
    real_samples = []
    k = 0
    for batch_condition, batch in dataset_train:
        batch_size = batch.shape[0]
        noise = tf.random.normal([batch_size, latent_dim, batch.shape[2]])
        gen_sample = generator_model([noise, batch_condition])
        for i in range(gen_sample.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples.append(gen_sample[i, -1, :])
            real_samples.append(batch[i, -1, :])
        k += 1
    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)

    generated_samples = generated_samples.reshape(generated_samples.shape[0], generated_samples.shape[1])
    real_samples = real_samples.reshape(real_samples.shape[0], real_samples.shape[1])
    np.save(f'../data/generated_samples_{prev_job_id}.npy', generated_samples)
    np.save(f'../data/real_samples_{prev_job_id}.npy', real_samples)
    # _, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    # plot_acf(generated_samples[:, 0], lags=20, ax=axes[0])
    # plot_acf(real_samples[:, 0], lags=20, ax=axes[1])
    # plt.savefig(f'plots/{prev_job_id}.pbs01_dense_dense_3_3_50_original/acf_day{N}.png')
    # plt.close()
    logging.info('Plotting the first 2 principal components of the generated and real samples...')
    plot_pca_with_marginals(generated_samples, real_samples, prev_job_id, args)
    logging.info('Done.')

    logging.info('Computing the average LOB shape')
    features = ['ask_volume_1', 'bid_volume_1', 'ask_volume_2', 'bid_volume_2', 'ask_volume_3', 'bid_volume_3']
    means_real, means_gen, p_values = [], [], []
    for i in range(n_features_gen):
        d_gen = np.round(generated_samples[:, i].flatten())
        d_real = real_samples[:, i].flatten()
        _, p_value = ttest_ind(d_gen, d_real, equal_var=False)
        means_real.append(np.mean(d_real))
        means_gen.append(np.mean(d_gen))
        p_values.append(p_value)

    width = 0.4  # Width of the bars
    indices = np.arange(len(features))  # Create indices for the x position
    indices1 = np.concatenate((np.arange(1,len(features),2)[::-1], np.arange(0,len(features),2)))
    print(np.array(features)[indices1])

    data = list(zip(features, p_values))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].bar(indices - width/2, np.array(means_gen)[indices1], width=width, label='Generated', alpha=0.85)  # Adjust x position for 'Generated' bars
    axes[0].bar(indices + width/2, np.array(means_real)[indices1], width=width, label='Real', alpha=0.85)  # Adjust x position for 'Real' bars
    axes[0].set_title(f'Average_LOB_shape')
    axes[0].set_xlabel('Levels')
    axes[0].set_xticks(indices)  # Set the x-tick labels to your features
    axes[0].set_xticklabels(np.array(features)[indices1], rotation=60)  # Rotate the x-tick labels by 90 degrees
    axes[0].legend()
    # Remove axes
    axes[1].axis('tight')
    axes[1].axis('off')
    # Create table and add it to the plot
    data = [(feat, f"{p:.2e}") for feat, p in data] # modify data to contain numbers in exponential form with 2 decimal positions
    axes[1].table(cellText=data, cellLoc='center', loc='center')
    # Add a title to the table
    axes[1].set_title(f"Welch's t-test p-values")
    plt.savefig(f'plots/{prev_job_id}.pbs01_dense_dense_3_3_50_original/average_LOB_shape.png')
    plt.close()
    logging.info('Done.')

    logging.info('Computing the errors on the correlation matrix using bootstrap...')
    noises = tf.random.normal([1, batch_size, latent_dim, input_train.shape[2]])
    correlation_matrix(dataset_train, generator_model, noises, T_gen, n_features_gen, prev_job_id, bootstrap_iterations=5000)
    logging.info('Done.')
    

