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
from scipy.stats import normaltest
from scipy.optimize import minimize

def rename_columns(dataframes_folder_path):
    '''This function takes as input the path of the folder containing the dataframes and renames the columns
    of the message and orderbook dataframes.
    
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
    n = orderbook.shape[1]
    bid_prices = np.array(orderbook[[f'Bid price {x}' for x in range(1, int(n/4)+1)]])
    bid_volumes = np.array(orderbook[[f'Bid size {x}' for x in range(1, int(n/4)+1)]])
    ask_prices = np.array(orderbook[[f'Ask price {x}' for x in range(1, int(n/4)+1)]])
    ask_volumes = np.array(orderbook[[f'Ask size {x}' for x in range(1, int(n/4)+1)]])
    return bid_prices, bid_volumes, ask_prices, ask_volumes


def preprocessing_orderbook_df(orderbook, message, sampling_freq, discard_time, n_levels):
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
    # Sample the orderbook every f events and take only the first n_levels levels
    orderbook = orderbook.iloc[::sampling_freq, :2*n_levels]
    # Reset the index
    orderbook = orderbook.reset_index(drop=True)
    return orderbook

def lob_reconstruction(N, tick, bid_prices, bid_volumes, ask_prices, ask_volumes):
    n_columns = bid_prices.shape[1]
    m, M = bid_prices.min().min(), ask_prices.max().max()
    for event in tqdm(range(N), desc='Computing LOB snapshots'):
        # Create the price and volume arrays
        p_line = np.arange(m, M+tick, tick)
        volumes = np.zeros_like(p_line)

        # Create two dictionaries to store the bid and ask prices keys and volumes as values
        d_ask = {ask_prices[event][i]: ask_volumes[event][i] for i in range(int(n_columns))}
        d_bid = {bid_prices[event][i]: -bid_volumes[event][i] for i in range(int(n_columns))}

        # Create two boolean arrays to select the prices in the p_line array that are also in the bid and ask prices
        mask_bid, mask_ask = np.in1d(p_line, list(d_bid.keys())), np.in1d(p_line, list(d_ask.keys()))

        # Assign to the volumes array the volumes corresponding to the the bid and ask prices
        volumes[np.where(mask_bid)] = list(d_bid.values())
        volumes[np.where(mask_ask)] = list(d_ask.values())
    
    return p_line, volumes


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
    if vector.shape[0] % N != 0:
        subvectors.append(vector[(i+1)*length:])
    return subvectors

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
    if len(outputs) == 2:
        real_output, fake_output = outputs
        real_accuracy = tf.reduce_mean(tf.cast(tf.greater_equal(real_output, 0.5), tf.float32))
        fake_accuracy = tf.reduce_mean(tf.cast(tf.less(fake_output, 0.5), tf.float32))
        total_accuracy = (real_accuracy + fake_accuracy) / 2.0
    else:
        fake_output = outputs
        total_accuracy = tf.reduce_mean(tf.cast(tf.less(fake_output, 0.5), tf.float32))
    return total_accuracy

def explore_latent_space_loss(candidate_noise, x_target, T_condition, generator):
    candidate_gen = generator([candidate_noise, x_target[:T_condition, :]])
    candidate_gen = tf.reshape(candidate_gen, [candidate_gen.shape[1], candidate_gen.shape[2]]).numpy()
    corr = np.corrcoef(x_target[T_condition:, :], candidate_gen)[0, 1] # corrcoef returns a matrix
    loss = 1 - corr
    return loss

def explore_latent_space(x_target, latent_dim, T_condition, T_gen, generator):
    logging.info('Exploring the latent space...')
    logging.info(f'x_target:\n\t{x_target}')
    logging.info(f'x_target.shape:\n\t{x_target.shape}')
    candidate_noise = np.random.normal(size=(1 , T_gen*latent_dim, x_target.shape[0]))
    logging.info(f'candidate_noise:\n\t{candidate_noise}')
    result = minimize(explore_latent_space_loss, candidate_noise, args=(x_target, T_condition, generator), method='L-BFGS-B')
    logging.info(f'result:\n\t{result}')
    return result.x


def plot_samples(dataset, number_of_batches_plot, generator_model, features, T_real, T_condition, latent_dim, n_features_input, job_id, epoch, scaler, args, final=False):
    '''This function plots the generated samples, together with the real one and the empirical distribution of the generated samples.'''
    if final:
        counter = f'{epoch}_final'
    else:
        counter = epoch
    # Select randomly an index to start from
    idx = np.random.randint(0, len(dataset)-number_of_batches_plot-1)
    dataset = dataset.skip(idx)

    # Check if the dataset is conditioned or not
    for batch in dataset.take(1):
        dim_batch = np.array(tf.shape(batch[0]))
    if dim_batch[1] > 1: 
        use_condition = True
    else:
        use_condition = False

    # Create two lists to store the generated and real samples. 
    # These list will be of shape (batch_size*number_of_batches_plot, T_real, n_features_input)
    generated_samples = []
    real_samples = []
    c = 0

    if use_condition == True:
        for batch_condition, batch in dataset:
            c += 1
            batch_size = batch.shape[0]
            noise = tf.random.normal([batch_size, T_real*latent_dim, n_features_input])
            # gen_sample is a tensor of shape (batch_size, T_real, n_features_input)
            # The generator model outputs a number of samples equal to the batch size
            gen_sample = generator_model([noise, batch_condition], training=True)
            batch = scaler.inverse_transform(tf.reshape(batch, [batch.shape[0], batch.shape[1]*batch.shape[2]])).reshape(batch.shape)
            gen_sample = scaler.inverse_transform(tf.reshape(gen_sample, [gen_sample.shape[0], gen_sample.shape[1]*gen_sample.shape[2]])).reshape(gen_sample.shape)
            # Append each sample to the lists
            for i in range(gen_sample.shape[0]):
                # All the appended samples will be of shape (T_real, n_features_input)
                generated_samples.append(gen_sample[i, :, :])
                real_samples.append(batch[i, :, :])
            if c == number_of_batches_plot: break
    else:
        for batch in dataset:
            c += 1
            batch_size = batch.shape[0]
            noise = tf.random.normal([batch_size, (T_real+T_condition)*latent_dim, n_features_input])
            gen_sample = generator_model(noise, training=True)
            batch = scaler.inverse_transform(tf.reshape(batch, [batch.shape[0], batch.shape[1]*batch.shape[2]])).reshape(batch.shape)
            gen_sample = scaler.inverse_transform(tf.reshape(gen_sample, [gen_sample.shape[0], gen_sample.shape[1]*gen_sample.shape[2]])).reshape(gen_sample.shape)
            for i in range(gen_sample.shape[0]):
                generated_samples.append(gen_sample[i, :, :])
                real_samples.append(batch[i, :, :])
            if c == number_of_batches_plot: break
    generated_samples = np.array(generated_samples)
    real_samples = np.array(real_samples)
    
    # features = [f'Curve{i+1}' for i in range(n_features_input)]
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
        fig, axes = plt.subplots(n_features_input, 1, figsize=(10, 10), tight_layout=True)
        fig1, axes1 = plt.subplots(n_features_input, 1, figsize=(10, 10), tight_layout=True)
        for i, feature in enumerate(features):
            # if feature == 'Price':
            #     d_gen = np.diff(generated_samples[:, :, i].flatten()/100)
            #     d_real = np.diff(real_samples[:, :, i].flatten()/100)
            # else:
            d_gen = generated_samples[:, :, i].flatten()
            d_real = real_samples[:, :, i].flatten()
            axes[i].plot(d_gen[:200], label='Generated', alpha=0.85)
            axes[i].plot(d_real[:200], label='Real', alpha=0.85)
            axes[i].set_title(f'Generated {feature}_{epoch}')
            axes[i].legend()
            axes1[i].hist(d_gen, bins=100, label='Generated', alpha=0.3)
            axes1[i].hist(np.round(d_gen), bins=100, label='Generated rounded', alpha=0.85)
            axes1[i].hist(d_real, bins=100, label='Real', alpha=0.85)
            axes1[i].set_title(f'Generated {feature}_{epoch}')
            axes1[i].legend()
        axes[i].set_xlabel('Time (Events)')
        axes1[i].set_xlabel('Values')
    path = f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.data}_{args.T_condition}_{args.loss}'
    fig.savefig(f'{path}/003_generated_samples_{counter}.png')
    fig1.savefig(f'{path}/004_generated_samples_hist_{counter}.png')
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

def anomaly_injection(message_df, anomaly_type):
    '''This function takes as input a message dataframe and injects anomalies in it. There will be
    two types of anomalies: the first one is the random submissions of enormous orders (buy or sell)
    at a random price, while the second one consists in filling one side of the LOB with a lot of orders.'''
    # Select the 1% of the data to inject anomalies
    n_anomalies = int(len(message_df) * 0.01)
    if anomaly_type == 'big_order':
        # Select randomly the indices where to inject the big orders
        indices = np.random.choice(len(message_df), n_anomalies, replace=False)
        message_df.loc[indices]['Event type'] = 4
        message_df.loc[indices]['Size'] = 100000
    if anomaly_type == 'fill_side':
        # Select randomly 10 consecutive indices where to inject the orders
        index = np.random.choice(len(message_df)-11, 1)
        message_df.loc[index: index+10]['Event type'] = 4
        message_df.loc[index: index]['Size'] = 500
        message_df.loc[index : index]['Direction'] = 1
    return message_df



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

    

