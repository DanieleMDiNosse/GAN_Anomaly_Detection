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

    # Set the seed for reproducibility
    np.random.seed(666)

    # Print the current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Create the folder for the plots
    if not os.path.exists('plots/anomaly_results'):
        os.makedirs('plots/anomaly_results')
    if not os.path.exists(f'plots/anomaly_results/{job_id}'):
        os.makedirs(f'plots/anomaly_results/{job_id}')

    # Days used in the training process
    N = args.N_days
    
    # Load the model
    path = f'models/{args.job_id}.pbs01_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}'
    gen_model_path = [p for p in os.listdir(path) if 'generator_model' in p]
    disc_model_path = [p for p in os.listdir(path) if 'discriminator_model' in p]
    generator = tf.keras.models.load_model(f'{path}/{gen_model_path[0]}')
    discriminator = tf.keras.models.load_model(f'{path}/{disc_model_path[0]}')
    logging.info(f'Loaded models from {path}')
    logging.info('Generator summary:')
    generator.summary(print_fn=lambda x: logging.info(x))
    logging.info('Discriminator summary:')
    discriminator.summary(print_fn=lambda x: logging.info(x))

    window_size = args.T_condition + 10
    depth = args.depth

    # Create the orderbook dataframe
    orderbook_df = create_LOB_snapshots(N, previos_days=False)
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
    logging.info(f'\nAre anomaly_data and normal_data the same?\n\t{np.all(anomaly_data.flatten() == normal_data.flatten())}')
    # check where they differ
    logging.info(f'\nWhere do anomaly_data and normal_data differ?\n\t{np.where(anomaly_data != normal_data)}')

    # Take the anomaly data
    condition_test_anomaly, input_test_anomaly = anomaly_data[:, :args.T_condition, :], anomaly_data[:, args.T_condition:, :(2*depth)]

    logging.info(f'\ninput_test_anomaly shape:\n\t{input_test_anomaly.shape}')
    logging.info(f'condition_test_anomaly shape:\n\t{condition_test_anomaly.shape}')
    logging.info(f'input_test shape:\n\t{input_test.shape}')
    logging.info(f'condition_test shape:\n\t{condition_test.shape}')

    # Plot side by side the normal and the anomaly data corresponding to one of the chosen windows
    titles = np.array([[f'ask_volume_{i}', f'bid_volume_{i}'] for i in range(1, depth+1)]).flatten()
    # bid_titles = np.array([[f'bid_volume_{i}'] for i in range(1, 3+1)]).flatten()
    # titles = np.concatenate([ask_titles, bid_titles])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    for i in range(2*depth):
        if np.all(input_test[chosen_windows[0], :, i] == input_test_anomaly[chosen_windows[0], :, i]) == False:
            axs[0].plot(input_test[chosen_windows[0], :, i], label=f'{titles[i]}', alpha=0.75)
            axs[0].legend()
            axs[0].set_title('Normal')
            axs[1].plot(input_test_anomaly[chosen_windows[0], :, i], label=f'{titles[i]}', alpha=0.75)
            axs[1].legend()
            axs[1].set_title('Anomaly')
    plt.savefig(f'plots/anomaly_results/{job_id}/anomaly_normal_vs_anomaly_{args.anomaly_type}_{args.ampl}.png')
    plt.close()


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
    std_normal_scores, std_anomaly_scores = np.zeros(len(chosen_windows)), np.zeros(len(chosen_windows))

    # Create a figure outside the loop
    # fig, ax = plt.subplots(figsize=(10, 5))

    # Store the combined scores and labels for all iterations
    df_normal = pd.DataFrame(columns=['Scores', 'Anomalies', 'Type'])
    df_anomaly = pd.DataFrame(columns=['Scores', 'Anomalies', 'Type'])
    for idx in range(len(chosen_windows)):
        input_real = input_test[chosen_windows[idx], :, :]
        idx_windows = np.random.choice(condition_test.shape[0], 100, replace=False)
        condition_test_sampled = condition_test[idx_windows]
        disc_scores = []
        for i, cond_test in enumerate(condition_test_sampled):
            disc_output = discriminator([input_real[None, ...], cond_test[None, ...]])
            df_normal.loc[(idx*100)+i] = [disc_output.numpy()[0][0], idx, 'Normal']
            disc_scores.append(disc_output.numpy()[0][0]) # The (1 - ) sign is due to the flipping of labels in the discriminator during training

        input_anomaly = input_test_anomaly[chosen_windows[idx], :, :]
        condition_anomaly_sampled = condition_test_anomaly[idx_windows]
        disc_scores_anomaly = []
        for i, cond_test_anomaly in enumerate(condition_anomaly_sampled):
            disc_output = discriminator([input_anomaly[None, ...], cond_test_anomaly[None, ...]])
            df_anomaly.loc[(idx*100)+i] = [disc_output.numpy()[0][0], idx, 'Anomaly']
            disc_scores_anomaly.append(disc_output.numpy()[0][0])

        # Collect the standard deviations of the scores
        std_normal_scores[idx] = np.std(disc_scores)
        std_anomaly_scores[idx] = np.std(disc_scores_anomaly)

    df = pd.concat([df_normal, df_anomaly])

    plt.figure(figsize=(10, 5), tight_layout=True)
    # Plot the combined scores with hue
    sns.violinplot(data=df, x='Anomalies', y='Scores', hue='Type', split=True, inner='quart', gap=0.8, palette='pastel')

    # Set the title of the plot
    plt.ylabel('Discriminator Scores')

    # Save the figure
    plt.savefig(f'plots/anomaly_results/{job_id}/discriminator_scores_combined_all_{args.anomaly_type}_{args.ampl}.png')
    plt.close()

    # Create the datsasets 
    dataset_normal = tf.data.Dataset.from_tensor_slices((condition_test, input_test)).batch(batch_size)
    dataset_anomaly = tf.data.Dataset.from_tensor_slices((condition_test_anomaly, input_test_anomaly)).batch(batch_size)

    disc_normal_score = []
    disc_anomaly_score = []
    # iterate through the dataset_normal and dataset_anomaly

    for (condition_real, input_real), (condition_anomaly, input_anomaly) in zip(dataset_normal, dataset_anomaly):
        disc_output_normal = discriminator([input_real, condition_real])
        # The (1 -) is due to the flipping of labels in the discriminator during training
        disc_normal_score.append(disc_output_normal.numpy()[0])
        disc_output_anomaly = discriminator([input_anomaly, condition_anomaly])
        disc_anomaly_score.append(disc_output_anomaly.numpy()[0])
    
    disc_normal_score = np.array(disc_normal_score)
    disc_anomaly_score = np.array(disc_anomaly_score)
    logging.info('Done')

    # Plot the normal scores
    v_idxs = np.zeros_like(disc_normal_score)
    v_idxs[chosen_windows] = disc_normal_score[chosen_windows]
    alpha = [1 if i != 0 else 0 for i in v_idxs]
    stds_normal = np.zeros_like(disc_normal_score)
    stds_normal[chosen_windows] = std_normal_scores.reshape(-1, 1)
    stds_normal = stds_normal.reshape(stds_normal.shape[0])
    logging.info(f'stds_normal shape: {stds_normal.shape}')
    logging.info(f'v_idxs shape: {v_idxs.shape}')

    _, axes = plt.subplots(2, 1, figsize=(10, 10), tight_layout=True)
    axes[0].plot(disc_normal_score, label='Discriminator score', alpha=0.7)
    axes[0].scatter(np.arange(disc_normal_score.shape[0]), v_idxs, color='red', alpha=alpha)
    axes[0].errorbar(chosen_windows, v_idxs.reshape(v_idxs.shape[0])[chosen_windows],
                yerr=stds_normal[chosen_windows],
                fmt='none', ecolor='red', color='red', capsize=5, alpha=0.6)
    axes[0].set_title('Anomaly scores')

     # Plot the anomaly scores
    v_idxs = np.zeros_like(disc_anomaly_score)
    v_idxs[chosen_windows] = disc_anomaly_score[chosen_windows]
    stds_anomaly = np.zeros_like(disc_anomaly_score)
    stds_anomaly[chosen_windows] = std_anomaly_scores.reshape(-1,1)
    stds_anomaly = stds_anomaly.reshape(stds_anomaly.shape[0])

    axes[1].plot(disc_anomaly_score, label='Discriminator score', alpha=0.7)
    axes[1].scatter(np.arange(disc_anomaly_score.shape[0]), v_idxs, color='red', label='Injected anomaly', alpha=alpha)
    axes[1].errorbar(chosen_windows, v_idxs.reshape(v_idxs.shape[0])[chosen_windows],
                yerr=stds_anomaly[chosen_windows],
                fmt='none', ecolor='red', color='red', capsize=5, alpha=0.6)
    axes[1].set_title('Anomaly scores (injected anomalies)')
    plt.savefig(f'plots/anomaly_results/{job_id}/anomaly_scores_{args.anomaly_type}_{args.ampl}.png')