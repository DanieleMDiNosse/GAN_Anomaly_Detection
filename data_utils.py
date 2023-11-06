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
import tensorflow as tf
from PIL import Image, ImageSequence
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import normaltest, ttest_ind
import math
from scipy.optimize import minimize

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
        start_ask = ask_prices[row, 0]
        start_bid = bid_prices[row, 0]
        volume_ask[row, 0] = ask_volumes[row, 0] # volume at best
        volume_bid[row, 0] = bid_volumes[row, 0] # volume at best
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


def plot_samples(dataset, generator_model, noises, features, T_gen, n_features_gen, job_id, epoch, scaler, args):
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
    scaler : sklearn scaler
        Scaler used to scale the data. It can be None.
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
        gen_sample = generator_model([noises[k], batch_condition])
        if scaler == None:
            gen_sample = transform_and_reshape(gen_sample, T_gen, n_features_gen)
            batch = transform_and_reshape(batch, T_gen, n_features_gen)
        else:
            batch = scaler.inverse_transform(tf.reshape(batch, [batch.shape[0], batch.shape[1]*batch.shape[2]])).reshape(batch.shape)
            gen_sample = scaler.inverse_transform(tf.reshape(gen_sample, [gen_sample.shape[0], gen_sample.shape[1]*gen_sample.shape[2]])).reshape(gen_sample.shape)
        for i in range(gen_sample.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples.append(gen_sample[i, :, :])
            real_samples.append(batch[i, :, :])
        k += 1

    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)
    
    means_real, means_gen, p_values = [], [], []
    if len(features) == 1:
        generated_samples = generated_samples.flatten()
        real_samples = real_samples.flatten()
        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.plot(generated_samples, label='Generated', alpha=0.85)
        plt.plot(real_samples, label='Real', alpha=0.85)
        plt.title(f'Generated {features[0]}_{epoch}')
        plt.xlabel('Time (Events)')
        plt.legend()
    else:
        fig, axes = plt.subplots(n_features_gen, 1, figsize=(10, 10), tight_layout=True)
        fig1, axes1 = plt.subplots(n_features_gen, 1, figsize=(10, 10), tight_layout=True)
        for i, feature in enumerate(features):
            if feature == 'Price':
                d_gen = simple_rounding(generated_samples[:, :, i].flatten())
                d_real = simple_rounding(real_samples[:, :, i].flatten())
            else:
                d_gen = np.round(generated_samples[:, :, i].flatten())
                d_real = real_samples[:, :, i].flatten()
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
    axes[1].table(cellText=data, cellLoc='center', loc='center')
    # Add a title to the table
    axes[1].set_title(f"Welch's t-test p-values_{epoch}")
    plt.savefig(f'{path}/5_average_LOB_shape_{epoch}.png')

    generated_samples = generated_samples.reshape(generated_samples.shape[0]*generated_samples.shape[1], generated_samples.shape[2])
    real_samples = real_samples.reshape(real_samples.shape[0]*real_samples.shape[1], real_samples.shape[2])
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

def correlation_matrix(dataset, generator_model, noises, best_epoch, scaler, T_gen, n_features_gen, job_id, bootstrap_iterations=1000):
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
        Number of bootstrap iterations. The default is 1000.
    
    Returns
    -------
    None.'''
    generated_samples = []
    real_samples = []
    k = 0
    for batch_condition, batch in dataset:
        gen_sample = generator_model([noises[k], batch_condition])
        if scaler == None:
            gen_sample = transform_and_reshape(gen_sample, T_gen, n_features_gen)
            batch = transform_and_reshape(batch, T_gen, n_features_gen)
        else:
            batch = scaler.inverse_transform(tf.reshape(batch, [batch.shape[0], batch.shape[1]*batch.shape[2]])).reshape(batch.shape)
            gen_sample = scaler.inverse_transform(tf.reshape(gen_sample, [gen_sample.shape[0], gen_sample.shape[1]*gen_sample.shape[2]])).reshape(gen_sample.shape)
        for i in range(gen_sample.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples.append(gen_sample[i, :, :])
            real_samples.append(batch[i, :, :])
        k += 1
    
    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)
    generated_samples = generated_samples.reshape(generated_samples.shape[0]*generated_samples.shape[1], generated_samples.shape[2])
    real_samples = real_samples.reshape(real_samples.shape[0]*real_samples.shape[1], real_samples.shape[2])

    generated_samples_bootstrap = np.copy(generated_samples)
    real_samples_bootstrap = np.copy(real_samples)

    correlations_gen = np.zeros(shape=(bootstrap_iterations, generated_samples.shape[1], generated_samples.shape[1]))
    correlations_real = np.zeros(shape=(bootstrap_iterations, generated_samples.shape[1], generated_samples.shape[1]))
    for i in range(bootstrap_iterations):
        logging.info(f'Bootstrap iteration {i+1}/{bootstrap_iterations}')
        # Shuffle randomly the generated samples and the real samples
        np.random.shuffle(generated_samples_bootstrap)
        np.random.shuffle(real_samples_bootstrap)
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
    
    logging.info(f'standard_deviation_gen shape: {standard_deviations_gen.shape}')
    logging.info(f'correlation_matrix_gen shape: {correlation_matrix_gen.shape}')

    correlation_matrix_gen = np.corrcoef(generated_samples, rowvar=False)
    correlation_matrix_real = np.corrcoef(real_samples, rowvar=False)
    features = np.array([[f'ask_volume_{i}',f'bid_volume_{i}'] for i in range(1, int(0.5*generated_samples.shape[1])+1)]).flatten()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
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
            text_str = f"{round(correlation_matrix_gen[i, j], 2)} ± {round(2*std_matrix_gen[i, j], 2)}"
            axes[0].text(j, i, text_str,
                ha='center', va='center',
                color='black', fontsize=8)
            k += 1
    axes[1].imshow(correlation_matrix_real, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title('Correlation Matrix (real samples)')
    axes[1].set_xticks(range(generated_samples.shape[1]))
    axes[1].set_yticks(range(generated_samples.shape[1]))
    axes[1].set_xticklabels(features, rotation=90)
    axes[1].set_yticklabels(features)
    for i in range(correlation_matrix_real.shape[0]):
        for j in range(correlation_matrix_real.shape[1]):
            text_str = f"{round(correlation_matrix_real[i, j], 2)} ± {round(2*std_matrix_real[i, j], 2)}"
            axes[1].text(j, i, text_str,
                    ha='center', va='center',
                    color='black', fontsize=8)
    path = [s for s in os.listdir('plots/') if job_id in s][0]
    plt.savefig(f'plots/{path}/6_correlation_matrix_final.png')
    plt.close()
    return None

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

def anomaly_injection_orderbook(fp, means, stds, T_condition, depth, std_ampl, anomaly_type):
    '''This function takes as input a orderbook dataframe and injects anomalies in it. There are
    two types of anomalies: the first one is the random submissions of enormous orders (buy or sell)
    at a random level, while the second one consists in filling one side of the LOB with a lot of orders.
    The anomalies are injected in the 0.5% of the windows. An anomaly is set as the mean of the level
    plus/minus the standard deviation of the level multiplied by a constant std_ampl.
    
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
    std_ampl : float
        Constant used to amplify the standard deviation.
    anomaly_type : string
        Type of anomaly to inject. It can be 'big_order' or 'fill_side'.
    
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
        chosen_timestamp = np.random.randint(T_condition, data.shape[1])
        chosen_windows.append(chosen_window)

        if anomaly_type == 'big_order':
            chosen_feature = np.random.randint(0, (2*depth))
            logging.info(f'Chosen feature: {chosen_feature}')
            if chosen_feature % 2 == 0:
                anomaly = means[chosen_feature] + stds[chosen_feature]*std_ampl
            else:
                anomaly = means[chosen_feature] - stds[chosen_feature]*std_ampl
            data[chosen_window, chosen_timestamp, chosen_feature] = anomaly

        if anomaly_type == 'fill_side':
            random_number_1 = np.random.normal()
            if random_number_1 < 0:
                bid_idxs = np.arange(1, fp.shape[2], 2)
                data[chosen_window, chosen_timestamp, bid_idxs] = means[bid_idxs] - stds[bid_idxs]*std_ampl
            else:
                ask_idxs = np.arange(0, fp.shape[2], 2)
                data[chosen_window, chosen_timestamp, ask_idxs] = means[ask_idxs] + stds[ask_idxs]*std_ampl
    return data, chosen_windows

def compute_spread(orderbook):
    '''This function computes the spread of the orderbook dataframe.'''
    spread = orderbook['Ask price 1'] - orderbook['Bid price 1']
    return spread


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This script several functions used to pre and post process the data.''')
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

    

