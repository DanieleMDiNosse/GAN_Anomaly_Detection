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
        description='''This script is aimed to preprocess the data for model testing. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument("-j", "--job_id", type=str, help=("Provide job id"))
    parser.add_argument('-at', '--anomaly_type', type=str, help='Type of anomaly to inject (big_order, fill_side)')
    parser.add_argument('-N', '--N_days', help='Number of days used for testing', type=int, default=1)
    parser.add_argument('-d', '--depth', help='Number of levels to consider in the orderbook', type=int)
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of generator layers', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of discriminator layers', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-tc', '--T_condition', help='Number of time steps to condition on', type=int)
    parser.add_argument('-ls', '--loss', help='Loss function (original, wasserstein)', type=str, default='original')
    parser.add_argument('-std', '--std_ampl', help='Amplification factor for the standard deviation', type=float, default=1)


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

    # Days used in the training process
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
    total_depth = 5
    window_size = args.T_condition + 10
    depth = args.depth

    # Read the dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    orderbook_df_paths = [path for path in dataframes_paths if 'orderbook' in path]
    orderbook_df_paths.sort()
    orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][N:N+1]
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()
    message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][N:N+1]

    # Preprocess the data using preprocessing_orderbook_df
    orderbook_dfs, discard_times_list = zip(*[(preprocessing_orderbook_df(df, msg, discard_time=1800)) for df, msg in zip(orderbook_dfs, message_dfs)])
    # Merge all the dataframes into a single one
    orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
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
    # Compute the first two moments
    means, stds = orderbook_df.mean(), orderbook_df.std()

    data_input = orderbook_df.values

    # Divide input data into overlapping pieces
    sub_data, length = divide_into_overlapping_pieces(data_input, window_size, 5)

    if sub_data[-1].shape[0] < window_size:     
        raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

    final_shape = (length-5*(window_size-1), window_size, orderbook_df.shape[1])
    # fp = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape)
    normal_data = np.empty(final_shape, dtype='float32')

    start_idx = 0
    for piece_idx, data in enumerate(sub_data):
        logging.info(f'{piece_idx+1}/{5}')
        logging.info(f'data shape:\n\t {data.shape}')
        windows = np.array(divide_into_windows(data, window_size))
        logging.info(f'windows shape:\n\t {windows.shape}')
        end_idx = start_idx + windows.shape[0]
        normal_data[start_idx:end_idx] = windows
        start_idx = end_idx
        del windows  # Explicit deletion

    # Take the original data
    condition_test, input_test = normal_data[:, :args.T_condition, :], normal_data[:, args.T_condition:, :(2*depth)]

    # Inject anomalies
    anomaly_data, chosen_windows = anomaly_injection_orderbook(normal_data, means, stds, args.T_condition, depth, args.std_ampl, args.anomaly_type)
    logging.info(f'\nanomaly_data shape:\n\t{anomaly_data.shape}')
    logging.info(f'normal_data shape:\n\t{normal_data.shape}')
    logging.info(f'\nchosen_windows:\n\t{chosen_windows}')

    # check if fp_anomaly and normal_data are the same
    logging.info(f'\nAre anomaly_data and normal_data the same?\n\t{np.all(anomaly_data == normal_data)}')
    # check where they differ
    logging.info(f'\nWhere do anomaly_data and normal_data differ?\n\t{np.where(anomaly_data != normal_data)}')

    # Take the anomaly data
    condition_test_anomaly, input_test_anomaly = anomaly_data[:, :args.T_condition, :], anomaly_data[:, args.T_condition:, :(2*depth)]

    logging.info(f'\ninput_test_anomaly shape:\n\t{input_test_anomaly.shape}')
    logging.info(f'condition_test_anomaly shape:\n\t{condition_test_anomaly.shape}')
    logging.info(f'input_test shape:\n\t{input_test.shape}')
    logging.info(f'condition_test shape:\n\t{condition_test.shape}')

    # Plot side by side the normal and the anomaly data corresponding to one of the chosen windows
    titles = np.array([[f'ask_volume_{i} (normal)',f'bid_volume_{i} (anomaly)'] for i in range(1, depth+1)]).flatten()
    fig, axs = plt.subplots(2*depth, 2, figsize=(10, 10), tight_layout=True)
    for i in range(2*depth):
        if np.all(input_test[chosen_windows[0], :, i] == input_test_anomaly[chosen_windows[0], :, i]):
            axs[i, 0].plot(input_test[chosen_windows[0], :, i], 'k', alpha=0.75)
            axs[i, 0].set_title(titles[i])
            axs[i, 1].plot(input_test_anomaly[chosen_windows[0], :, i], 'k', alpha=0.75)
            axs[i, 1].set_title(titles[i])
        else:
            axs[i, 0].plot(input_test[chosen_windows[0], :, i], 'coral', alpha=0.75)
            axs[i, 0].set_title(titles[i])
            axs[i, 1].plot(input_test_anomaly[chosen_windows[0], :, i], 'coral', label='Anomaly', alpha=0.75)
            axs[i, 1].set_title(titles[i])
            axs[i, 1].legend()
    plt.savefig(f'output_{job_id}_normal_vs_anomaly_{args.anomaly_type}_{args.std_ampl}.png')


    # Hyperparameters
    n_features = input_test.shape[2]
    T_gen = input_test.shape[1]
    latent_dim = depth*10
    batch_size = 1

    logging.info('Computing the scores...')

    # Create the datsasets 
    dataset_normal = tf.data.Dataset.from_tensor_slices((condition_test, input_test)).batch(batch_size)
    dataset_anomaly = tf.data.Dataset.from_tensor_slices((condition_test_anomaly, input_test_anomaly)).batch(batch_size)

    disc_normal_score = []
    disc_anomaly_score = []
    # iterate through the dataset_normal and dataset_anomaly

    for (condition_real, input_real), (condition_anomaly, input_anomaly) in zip(dataset_normal, dataset_anomaly):
        # add one dummy dimension to window_condition due to contestts on the shape of the input for the generator
        # window_condition = np.expand_dims(window_condition, axis=0)
        # Find the latent space representation of the window
        # res = explore_latent_space(input_real, condition_real, latent_dim, T_gen, generator)
        # Use it to generate a sample
        # gen_output = generator(res)
        # gen_output = tf.reshape(gen_output, [T_gen, n_features]).numpy()
        # Compute the difference wrt the original window
        # diff = np.sum(np.abs(window_test - gen_output))
        # Compute the score given by the discriminator
        # Append the scores
        # gen_anomaly_scores.append(diff)
        disc_output_normal = discriminator([input_real, condition_real])
        logging.info(f'\nDiscriminator output normal:\n\t{disc_output_normal.numpy()[0]}')
        disc_normal_score.append(disc_output_normal.numpy()[0])
        disc_output_anomaly = discriminator([input_anomaly, condition_anomaly])
        logging.info(f'\nDiscriminator output anomaly:\n\t{disc_output_anomaly.numpy()[0]}')
        disc_anomaly_score.append(disc_output_anomaly.numpy()[0])
    
    # gen_anomaly_scores = np.array(gen_anomaly_scores)
    disc_normal_score = np.array(disc_normal_score)
    disc_anomaly_score = np.array(disc_anomaly_score)
    logging.info('Done')

    # Plot the normal scores
    v_idxs = np.zeros_like(disc_normal_score)
    v_idxs[chosen_windows] = disc_normal_score[chosen_windows]
    alpha = [1 if i != 0 else 0 for i in v_idxs]

    plt.figure()
    plt.plot(disc_normal_score, label='Discriminator score', alpha=0.7)
    plt.scatter(np.arange(disc_normal_score.shape[0]), v_idxs, color='red', alpha=alpha)
    # plt.ylim(disc_normal_score.min()-0.1, disc_anomaly_score.max()+0.1)
    plt.title('Anomaly scores')
    plt.savefig(f'output_{job_id}_anomaly_scores_normal_{args.anomaly_type}_{args.std_ampl}.png')

     # Plot the anomaly scores
    v_idxs = np.zeros_like(disc_anomaly_score)
    v_idxs[chosen_windows] = disc_anomaly_score[chosen_windows]

    plt.figure()
    plt.plot(disc_anomaly_score, label='Discriminator score', alpha=0.7)
    plt.scatter(np.arange(disc_anomaly_score.shape[0]), v_idxs, color='red', label='Injected anomaly', alpha=alpha)
    # plt.ylim(disc_anomaly_score.min()-0.1, disc_anomaly_score.max()+0.1)
    plt.title('Anomaly scores (injected anomalies)')
    plt.savefig(f'output_{job_id}_anomaly_scores_synthetic_{args.anomaly_type}_{args.std_ampl}.png')