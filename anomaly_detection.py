import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import pandas as pd
from data_utils import *
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 2 to avoid the messages about the compilation of the CUDA code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import logging
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to check the ability of the trained GAN to detect anomalies.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument("-j", "--job_id", type=str, help=("Provide job id"))
    parser.add_argument('-at', '--anomaly_type', type=str, help='Type of anomaly to inject (big_order, metaorder)')
    parser.add_argument('-N', '--N_days', help='Number of days used for testing', type=int, default=1)
    parser.add_argument('-d', '--depth', help='Number of levels to consider in the orderbook', type=int)
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of generator layers', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of discriminator layers', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-tc', '--T_condition', help='Number of time steps to condition on', type=int)
    parser.add_argument('-ls', '--loss', help='Loss function (original, wasserstein)', type=str, default='original')
    parser.add_argument('-a', '--ampl', help='Amplification factor for the standard deviation', type=float, default=1)


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
    logging.basicConfig(filename=f'anomaly_detection_{job_id}.log', format='%(message)s', level=levels[args.log])

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

    window_size = args.T_condition + 10
    depth = args.depth

    # Create the orderbook dataframe
    orderbook_df = create_orderbook_dataframe(N)
    # Compute the first two moments
    means, stds = orderbook_df.mean(), orderbook_df.std()

    data_input = orderbook_df.values

    # Divide input data into overlapping pieces
    sub_data, length = divide_into_overlapping_pieces(data_input, window_size, 5)

    if sub_data[-1].shape[0] < window_size:     
        raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

    final_shape = (length-5*(window_size-1), window_size, orderbook_df.shape[1])
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
    anomaly_data, chosen_windows, chosen_feature = anomaly_injection_orderbook(normal_data, args.T_condition, depth, args.ampl, args.anomaly_type)
    # Save the anomaly data
    np.save(f'anomaly_data_{N}.npy', anomaly_data)
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
    plt.savefig(f'anomaly_{job_id}_normal_vs_anomaly_{args.anomaly_type}_{args.ampl}.png')


    # Hyperparameters
    n_features = input_test.shape[2]
    T_gen = input_test.shape[1]
    latent_dim = depth*15
    batch_size = 1

    # The idea here is to create a sort of 'distribution of scores' for the normal and the anomaly data
    # to asses the ability of the discriminator to distinguish between the two. For each anomaly, I
    # sample a certain number of condition windows and compute the discriminator scores (normal and anomaly) for each of them.
    # Then, I create a violin plot of the distribution of the scores for the normal and the anomaly data.
    logging.info('Computing the scores...')
    # idx_windows = np.random.choice(len(chosen_windows), 10, replace=False)
    for idx in range(len(chosen_windows)):
        input_real = input_test[chosen_windows[idx], :, :]
        idx_windows = np.random.choice(condition_test.shape[0], 100, replace=False)
        condition_test_sampled = condition_test[idx_windows]
        disc_scores = []
        for cond_test in condition_test_sampled:
            disc_output = discriminator([input_real[None, ...], cond_test[None, ...]])
            disc_scores.append(disc_output.numpy()[0])

        input_anomaly = input_test_anomaly[chosen_windows[idx], :, :]
        condition_anomaly_sampled = condition_test_anomaly[idx_windows]
        disc_scores_anomaly = []
        for cond_test_anomaly in condition_anomaly_sampled:
            disc_output = discriminator([input_anomaly[None, ...], cond_test_anomaly[None, ...]])
            disc_scores_anomaly.append(disc_output.numpy()[0])

        # Combine the scores into one array and create a label array
        combined_scores = np.concatenate([np.ravel(disc_scores), np.ravel(disc_scores_anomaly)])
        labels = ['Normal'] * len(disc_scores) + ['Anomaly'] * len(disc_scores_anomaly)

        # Create a figure
        plt.figure(figsize=(10, 5))

        # Plot the combined scores with hue
        sns.violinplot(x=labels, y=combined_scores, split=True)

        # Set the title of the plot
        plt.ylabel('Discriminator Scores')

        # Save the figure
        plt.savefig(f'plots/discriminator_scores_combined_{idx}_{args.anomaly_type}.png')
        plt.close()

        # # Create a figure with two subplots
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # # Plot the normal scores in the first subplot
        # sns.violinplot(np.array(disc_scores), orientation='v', ax=axs[0])
        # axs[0].set_title('Normal Scores')
        # # Plot the anomaly scores in the second subplot
        # sns.violinplot(np.array(disc_scores_anomaly), orientation='v', ax=axs[1])
        # axs[1].set_title('Anomaly Scores')
        # # Set the title of the figure
        # fig.suptitle('Discriminator Scores')
        # # Save the figure
        # plt.savefig(f'plots/discriminator_scores_{idx}.png')

    

    # Create the datsasets 
    dataset_normal = tf.data.Dataset.from_tensor_slices((condition_test, input_test)).batch(batch_size)
    dataset_anomaly = tf.data.Dataset.from_tensor_slices((condition_test_anomaly, input_test_anomaly)).batch(batch_size)

    disc_normal_score = []
    disc_anomaly_score = []
    # iterate through the dataset_normal and dataset_anomaly

    for (condition_real, input_real), (condition_anomaly, input_anomaly) in zip(dataset_normal, dataset_anomaly):
        disc_output_normal = discriminator([input_real, condition_real])
        disc_normal_score.append(disc_output_normal.numpy()[0])
        disc_output_anomaly = discriminator([input_anomaly, condition_anomaly])
        disc_anomaly_score.append(disc_output_anomaly.numpy()[0])
    
    # gen_anomaly_scores = np.array(gen_anomaly_scores)
    disc_normal_score = np.array(disc_normal_score)
    disc_anomaly_score = np.array(disc_anomaly_score)
    logging.info('Done')

    # Plot the normal scores
    v_idxs = np.zeros_like(disc_normal_score)
    v_idxs[chosen_windows] = disc_normal_score[chosen_windows]
    alpha = [1 if i != 0 else 0 for i in v_idxs]

    _, axes = plt.subplots(2, 1, figsize=(10, 10), tight_layout=True)
    axes[0].plot(disc_normal_score, label='Discriminator score', alpha=0.7)
    axes[0].scatter(np.arange(disc_normal_score.shape[0]), v_idxs, color='red', alpha=alpha)
    # plt.ylim(disc_normal_score.min()-0.1, disc_anomaly_score.max()+0.1)
    axes[0].set_title('Anomaly scores')

     # Plot the anomaly scores
    v_idxs = np.zeros_like(disc_anomaly_score)
    v_idxs[chosen_windows] = disc_anomaly_score[chosen_windows]

    axes[1].plot(disc_anomaly_score, label='Discriminator score', alpha=0.7)
    axes[1].scatter(np.arange(disc_anomaly_score.shape[0]), v_idxs, color='red', label='Injected anomaly', alpha=alpha)
    # plt.ylim(disc_anomaly_score.min()-0.1, disc_anomaly_score.max()+0.1)
    axes[1].set_title('Anomaly scores (injected anomalies)')
    plt.savefig(f'anomaly_{job_id}_anomaly_scores_{args.anomaly_type}_{args.ampl}.png')