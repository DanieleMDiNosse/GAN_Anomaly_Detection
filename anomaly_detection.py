import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from data_utils import *
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 2 to avoid the messages about the compilation of the CUDA code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_utils import explore_latent_space, explore_latent_space_loss
import argparse
import logging
from joblib import load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument("-j", "--job_id", type=str, help=("Provide job id"))
    parser.add_argument('-N', '--N_days', help='Number of days used for training', type=int, default=1)
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of generator layers', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of discriminator layers', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-tc', '--T_condition', help='Number of time steps to condition on', type=int, default=2)
    parser.add_argument('-ls', '--loss', help='Loss function (original, wasserstein)', type=str, default='original')


    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    args = parser.parse_args()

    if os.getenv("PBS_JOBID") != None:
        job_id = os.getenv("PBS_JOBID")
    else:
        job_id = os.getpid()

    # Initialize logger
    logging.basicConfig(filename=f'output_{job_id}_anomalydet.log', format='%(message)s', level=levels[args.log])

    # Print the current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Days to consider
    N = args.N_days
    
    # Load the model
    path = f'models/{args.job_id}.pbs01_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}'
    gen_model_path = [p for p in os.listdir(path) if 'generator_model' in p]
    disc_model_path = [p for p in os.listdir(path) if 'discriminator_model' in p]
    generator = tf.keras.models.load_model(f'{path}/{gen_model_path[0]}')
    discriminator = tf.keras.models.load_model(f'{path}/{disc_model_path[0]}')
    logging.info('Generator summary:')
    generator.summary(print_fn=lambda x: logging.info(x))
    logging.info('Discriminator summary:')
    discriminator.summary(print_fn=lambda x: logging.info(x))

    # Load data
    stock = 'MSFT'
    date = '2018-04-01_2018-04-30_5'
    window_size = args.T_condition + 10
    depth = 1

    # Read the dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    orderbook_df_paths = [path for path in dataframes_paths if 'orderbook' in path]
    orderbook_df_paths.sort()
    orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][:N]
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()
    message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][:N]

    # Preprocess the data using preprocessing_orderbook_df
    orderbook_dfs, discard_times_list = zip(*[(preprocessing_orderbook_df(df, msg, discard_time=1800, sampling_freq=40, n_levels=depth)) for df, msg in zip(orderbook_dfs, message_dfs)])
    # Merge all the dataframes into a single one
    orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
    # Extract the prices and volumes
    bid_prices, bid_volumes, ask_prices, ask_volumes = prices_and_volumes(orderbook_df)
    # Compute the volumes considering also empty levels
    volumes_ask, volumes_bid = volumes_per_level(ask_prices, bid_prices, ask_volumes, bid_volumes, depth)
    # Create a dataframe with the volumes
    orderbook_df = pd.DataFrame()
    for i in range(depth):
        orderbook_df[f'volumes_ask_{i+1}'] = volumes_ask[:, i]
        orderbook_df[f'volumes_bid_{i+1}'] = volumes_bid[:, i]
    # Inject into the orderbook abnormal events
    orderbook_anomaly = anomaly_injection_orderbook(orderbook_df, anomaly_type='fill_side')
    # Normalize the data via sqrt
    orderbook_df = orderbook_anomaly.applymap(lambda x: math.copysign(1,x)*np.sqrt(np.abs(x))*0.1)

    data_input = orderbook_df.values

    # Divide input data into overlapping pieces
    sub_data, length = divide_into_overlapping_pieces(data_input, window_size, 5)

    if sub_data[-1].shape[0] < window_size:     
        raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

    final_shape = (length, window_size, orderbook_df.shape[1])
    fp = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape)

    start_idx = 0
    for piece_idx, data in enumerate(sub_data):
        logging.info(f'\t{piece_idx+1}/{5}')
        windows = np.array(divide_into_windows(data, window_size))
        logging.info(f'\twindows shape: {windows.shape}')
        end_idx = start_idx + windows.shape[0]
        fp[start_idx:end_idx] = windows
        start_idx = end_idx
        del windows  # Explicit deletion

    train, val = train_test_split(fp, train_size=0.75)
    del fp  # Explicit deletion
    condition_train_anomaly, input_train_anomaly = train[:, :args.T_condition, :], train[:, args.T_condition:, :]
    condition_val_anomaly, input_val_anomaly = val[:, :args.T_condition, :], val[:, args.T_condition:, :]

    np.save(f'../data/condition_train_{stock}_{window_size}_{N}days_orderbook_anomaly.npy', condition_train_anomaly)
    np.save(f'../data/condition_val_{stock}_{window_size}_{N}days_orderbook_anomaly.npy', condition_val_anomaly)
    np.save(f'../data/input_train_{stock}_{window_size}_{N}days_orderbook_anomaly.npy', input_train_anomaly)
    np.save(f'../data/input_val_{stock}_{window_size}_{N}days_orderbook_anomaly.npy', input_val_anomaly)

    input_train = np.load(f'../data/input_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
    input_val = np.load(f'../data/input_val_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
    condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
    condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
    
    logging.info(f'input_train shape:\n\t{input_train.shape}')
    logging.info(f'condition_train shape:\n\t{condition_train.shape}')
    logging.info(f'input_val shape:\n\t{input_val.shape}')
    logging.info(f'condition_val shape:\n\t{condition_val.shape}')


    # Hyperparameters
    n_features = input_train.shape[2]
    T_gen = input_train.shape[1]
    latent_dim = 10
    batch_size = 1

    # ---------------------------- Real Data ---------------------------- #
    # Create a datsaset from the input
    dataset = tf.data.Dataset.from_tensor_slices((condition_train, input_train)).batch(batch_size)

    gen_anomaly_scores = []
    disc_anomaly_score = []
    for window_condition, window_train in dataset:
        # add one dummy dimension to window_condition due to contraints on the shape of the input for the generator
        # window_condition = np.expand_dims(window_condition, axis=0)
        # Find the latent space representation of the window
        # res = explore_latent_space(window_train, window_condition, latent_dim, T_gen, generator)
        # Use it to generate a sample
        # gen_output = generator(res)
        # gen_output = tf.reshape(gen_output, [T_gen, n_features]).numpy()
        # Compute the difference wrt the original window
        # diff = np.sum(np.abs(window_train - gen_output))
        # Compute the score given by the discriminator
        disc_output = discriminator([window_train, window_condition])
        # Append the scores
        # gen_anomaly_scores.append(diff)
        disc_anomaly_score.append(disc_output.numpy()[0])
    
    # gen_anomaly_scores = np.array(gen_anomaly_scores)
    disc_anomaly_score = np.array(disc_anomaly_score)

    # Plot the anomaly scores
    plt.figure()
    plt.plot(gen_anomaly_scores, label='Generator score')
    plt.plot(disc_anomaly_score, label='Discriminator score')
    plt.legend()
    plt.savefig(f'output_{job_id}_anomaly_scores_normal.png')

    # ---------------------------- Anomaly Data ---------------------------- #
    # Create a datsaset from the input
    dataset = tf.data.Dataset.from_tensor_slices((condition_train_anomaly, input_train_anomaly)).batch(batch_size)

    gen_anomaly_scores = []
    disc_anomaly_score = []
    for window_condition, window_train in dataset:
        disc_output = discriminator([window_train, window_condition])
        disc_anomaly_score.append(disc_output.numpy()[0])
    disc_anomaly_score = np.array(disc_anomaly_score)

    plt.figure()
    plt.plot(gen_anomaly_scores, label='Generator score')
    plt.plot(disc_anomaly_score, label='Discriminator score')
    plt.legend()
    plt.savefig(f'output_{job_id}_anomaly_scores_synthetic.png')