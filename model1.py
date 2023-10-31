'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 2 to avoid the orderbooks about the compilation of the CUDA code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_utils import *
from sklearn.preprocessing import StandardScaler
import argparse
import logging
# from tensorflow.keras.utils import plot_model
from model_utils import *
from joblib import dump, load
import math
import gc
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
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
    parser.add_argument('-c', '--condition', action='store_true', help='Conditioning on the first T_condition time steps')
    parser.add_argument('-Tc', '--T_condition', help='Number of time steps to condition on', type=int, default=2)
    parser.add_argument('-Tg', '--T_gen', help='Number of time steps to generate', type=int, default=1)
    parser.add_argument('-ls', '--loss', help='Loss function (original, wasserstein)', type=str, default='original')

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

    logging.basicConfig(filename=f'output_{job_id}.log', format='%(message)s', level=levels[args.log])

    logger = tf.get_logger()
    logger.setLevel('ERROR')

    # Set the seed for TensorFlow to the number of the beast
    tf.random.set_seed(666)

    # Print the current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Enable device placement logging
    tf.debugging.set_log_device_placement(True)
   
    # Load the data
    # stock = 'TSLA'
    # date = '2015-01-01_2015-01-31_10'
    stock = 'MSFT'
    date = '2018-04-01_2018-04-30_5'
    total_depth = 5
    N = args.N_days
    depth = args.depth

    logging.info(f'Stock:\n\t{stock}')
    logging.info(f'Number of days:\n\t{N}')

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        logging.info("No GPUs available.")
    else:
        logging.info("Available GPUs:")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        for device in physical_devices:
            logging.info(f'\t{device}\n')
    
    # Folders creation
    os.mkdir(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Generated samples
    os.mkdir(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Models
    
    # Read the dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')

    orderbook_df_paths = [path for path in dataframes_paths if 'orderbook' in path]
    orderbook_df_paths.sort()
    orderbook_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in orderbook_df_paths][:N]

    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()
    message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][:N]

    # Preprocess the data using preprocessing_orderbook_df
    orderbook_dfs, discard_times_list = zip(*[(preprocessing_orderbook_df(df, msg, discard_time=1800, sampling_freq=40, n_levels=total_depth)) for df, msg in zip(orderbook_dfs, message_dfs)])
    logging.info(f'Discarded time (sod, eod):\n\t{[discard_times for discard_times in discard_times_list]}')
    # Merge all the dataframes into a single one
    orderbook_df = pd.concat(orderbook_dfs, ignore_index=True)
    # Evaluate the spread
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
    orderbook_df['spread'] = spread
    orderbook_df = orderbook_df.applymap(lambda x: math.copysign(1,x)*np.sqrt(np.abs(x))*0.1)

    # Define the parameters of the GAN. Some of them are set via argparse
    T_condition = args.T_condition
    T_gen = args.T_gen
    window_size = T_condition + T_gen
    n_features_input = orderbook_df.shape[1]
    n_features_gen = 2*depth
    latent_dim = args.latent_dim
    n_epochs = 10000
    batch_size = args.batch_size

    # Define the parameters for the early stopping criterion
    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 200

    # Scale all the data at once is prohibitive due to memory resources. For this reason, the data is divided into overlapping pieces
    # and each piece is scaled separately using the partial_fit function of StandardScaler. Note that this is the only scaler that
    # has this feature. Then, the scaled pieces are merged together and divided into windows.
    num_pieces = 5

    if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{N}days_orderbook_.npy'):
        logging.info('\n[Input] ---------- PREPROCESSING ----------')

        data_input = orderbook_df.values

        # Divide input data into overlapping pieces
        sub_data, length = divide_into_overlapping_pieces(data_input, window_size, num_pieces)

        if sub_data[-1].shape[0] < window_size:     
            raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

        logging.info(f'Number of windows: {length}')

        # # Create a scaler object to scale the data
        # scaler = StandardScaler()
        # logging.info(f'Memorize "sub" estimates of mean and variance...')
        # for piece_idx, data in enumerate(sub_data):
        #     logging.info(f'\t{piece_idx+1}/{num_pieces}')
        #     # The scaler is updated with the data of each piece
        #     scaler.partial_fit(data)
        # logging.info('Done.')

        # save the scaler
        # logging.info('Save the scaler...')
        # dump(scaler, f'scaler_{N}days_orderbook.joblib')
        # logging.info('Done.')

        # Create a memmap to store the scaled data.
        final_shape = (length, window_size, n_features_input)
        fp = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape)

        start_idx = 0
        logging.info(f'\nStart scaling the data...')
        for piece_idx, data in enumerate(sub_data):
            logging.info(f'\t{piece_idx+1}/{num_pieces}')
            # Here the scaling is performed and the resulting scaled data is assign divided into windows
            # and assigned to the memory mapped vector
            #scaled_data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
            windows = np.array(divide_into_windows(data, window_size))
            logging.info(f'\twindows shape: {windows.shape}')
            end_idx = start_idx + windows.shape[0]
            fp[start_idx:end_idx] = windows
            start_idx = end_idx
            del windows  # Explicit deletion
        logging.info('Done.')

        logging.info('\nSplit the condition data into train and validation sets...')
        train, val = train_test_split(fp, train_size=0.75)
        logging.info('Done.')

        if args.condition == True:
            logging.info('\nDividing each window into condition and input...')
            condition_train, input_train = train[:, :T_condition, :], train[:, T_condition:, :n_features_gen]
            condition_val, input_val = val[:, :T_condition, :], val[:, T_condition:, :n_features_gen]
            logging.info('Done.')

            logging.info(f'input_train shape:\n\t{input_train.shape}')
            logging.info(f'condition_train shape:\n\t{condition_train.shape}')
            logging.info(f'input_val shape:\n\t{input_val.shape}')
            logging.info(f'condition_val shape:\n\t{condition_val.shape}')

        logging.info('\nSave the files...')
        np.save(f'../data/condition_train_{stock}_{window_size}_{N}days_orderbook.npy', condition_train)
        np.save(f'../data/condition_val_{stock}_{window_size}_{N}days_orderbook.npy', condition_val)
        np.save(f'../data/input_train_{stock}_{window_size}_{N}days_orderbook.npy', input_train)
        np.save(f'../data/input_val_{stock}_{window_size}_{N}days_orderbook.npy', input_val)
        logging.info('Done.')

        logging.info('\n[Input] ---------- DONE ----------')
    else:
        logging.info('Loading input_train, input_validation and input_test sets...')
        input_train = np.load(f'../data/input_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
        input_val = np.load(f'../data/input_val_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
        condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
        condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
        
        logging.info(f'input_train shape:\n\t{input_train.shape}')
        logging.info(f'condition_train shape:\n\t{condition_train.shape}')
        logging.info(f'input_val shape:\n\t{input_val.shape}')
        logging.info(f'condition_val shape:\n\t{condition_val.shape}')
        
        logging.info('Loading the scaler...')
        scaler = load(f'scaler_{N}days_orderbook.joblib')
        logging.info('Done.')

    logging.info(f"\nHYPERPARAMETERS:\n"
                    f"\tstock: {stock}\n"
                    f"\tdepth: {depth}\n"
                    f"\tgenerator: {args.type_gen}\n"
                    f"\tdiscriminator: {args.type_disc}\n"
                    f"\tn_layers_gen: {args.n_layers_gen}\n"
                    f"\tn_layers_disc: {args.n_layers_disc}\n"
                    f"\tskip_connection: {args.skip_connection}\n"
                    f"\tlatent_dim per time: {latent_dim}\n"
                    f"\tn_features_input: {n_features_input}\n"
                    f"\tn_features_gen: {n_features_gen}\n"
                    f"\tfeatures: {orderbook_df.columns}\n"
                    f"\tn_epochs: {n_epochs}\n"
                    f"\tT_condition: {T_condition}\n"
                    f"\tT_gen: {T_gen}\n"
                    f"\tbatch_size: {batch_size} (num_batches: {input_train.shape[0]//batch_size})\n"
                    f"\tcondition: {args.condition}\n"
                    f"\tloss: {args.loss}\n"
                    f"\tpatience: {patience}\n"
                    f"\tjob_id: {job_id}\n")

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = [generator_optimizer, discriminator_optimizer]

    # Build the models
    if args.condition== True:
        generator_model = build_generator(args.n_layers_gen, args.type_gen, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, latent_dim, True)
        discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, True, args.loss)
        feature_extractor = build_feature_extractor(discriminator_model, [i for i in range(1, args.n_layers_disc)])
    else:
        generator_model = build_generator(args.n_layers_gen, args.type_gen, args.skip_connection, T_gen, T_condition, n_features_input, latent_dim, False)
        discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, args.skip_connection, T_gen, T_condition, n_features_input, False, args.loss)

    logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
    generator_model.summary(print_fn=logging.info)
    logging.info('\n')
    discriminator_model.summary(print_fn=logging.info)
    logging.info('[Model] ---------- DONE ----------\n')
    # exit()

    # Define a dictionary to store the metrics
    metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

    # Define checkpoint and checkpoint manager
    # job_id_restore = 192418
    checkpoint_prefix = f"models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    generator_model=generator_model,
                                    discriminator_model=discriminator_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=3)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    # # save the models
    # logging.info('Saving the models...')
    # generator_model.save(f'models/{job_id_restore}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.h5')
    # discriminator_model.save(f'models/{job_id_restore}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model.h5')
    # logging.info('Done.')
    # exit()


    # Train the GAN.
    logging.info('\n[Training] ---------- START TRAINING ----------')
    if args.condition == True:
        dataset_train = tf.data.Dataset.from_tensor_slices((condition_train, input_train)).batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices((condition_val, input_val)).batch(batch_size)
    else:
        dataset_train = tf.data.Dataset.from_tensor_slices((train)).batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices((val)).batch(batch_size)

    num_batches = len(dataset_train)
    logging.info(f'Number of batches:\n\t{num_batches}\n')

    # Initialize a list to store the mean over all the features of the wasserstein distances at each epoch
    wass_to_plot = []
    for epoch in range(n_epochs):
        j = 0
        noises = []
        if args.condition == True:
            for batch_condition, batch_real_samples in dataset_train:
                j += 1
                batch_size = batch_real_samples.shape[0]
                generator_model, discriminator_model, noise = train_step(batch_real_samples, batch_condition, generator_model, discriminator_model, feature_extractor, optimizer, args.loss, T_gen, T_condition, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, args.condition, args)
                noises.append(noise)
        else:
            for batch_real_samples in dataset_train:
                j += 1
                batch_condition = np.zeros_like(batch_real_samples)
                batch_size = batch_real_samples.shape[0]
                generator_model, discriminator_model = train_step(batch_real_samples, batch_condition, generator_model, discriminator_model, optimizer, args.loss, T_gen, T_condition, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, args.condition, args)
        # Save the models via checkpoint
        checkpoint_manager.save()

        if epoch % 10 == 0:
            logging.info('Creating a time series with the generated samples...')
            # Note that even you define number_of_batches_plot, you modified plot_sample in order to plot all the batches
            # number_of_batches_plot = 100 
            features = orderbook_df.columns[:n_features_gen]
            plot_samples(dataset_train, generator_model, noises, features, T_gen, T_condition, latent_dim, n_features_gen, job_id, epoch, None, args, final=False)
            logging.info('Done')

        if epoch > 1:
            logging.info('Check Early Stopping Criteria on Validation Set')

            wass_dist = [[] for _ in range(n_features_gen)]
            if args.condition == True:
                for val_batch_condition, val_batch in dataset_val:
                    batch_size = val_batch.shape[0]
                    noise = tf.random.normal([batch_size, T_gen*latent_dim, n_features_gen])
                    generated_samples = generator_model([noise, val_batch_condition], training=True)
                    for feature in range(n_features_gen):
                        for i in range(generated_samples.shape[0]):
                            w = wasserstein_distance(val_batch[i, :, feature], generated_samples[i, :, feature])
                            wass_dist[feature].append(w)
            else:
                for val_batch in dataset_val:
                    batch_size = val_batch.shape[0]
                    noise = tf.random.normal([batch_size, (T_gen+T_condition)*latent_dim, n_features_input])
                    generated_samples = generator_model(noise, training=True)
                    for feature in range(generated_samples.shape[2]):
                        for i in range(generated_samples.shape[0]):
                            w = wasserstein_distance(val_batch[i, :, feature], generated_samples[i, :, feature])
                            wass_dist[feature].append(w)

            # Compute the mean of all the wasserstein distances. The idea is to track this quantity
            # that is an aggregate measure of how the generator is able to generate samples that are
            # similar to the real ones.
            wass_dist = np.mean(np.array(wass_dist).flatten())
            wass_to_plot.append(wass_dist)

            logging.info(f'Wasserstein distance: {wass_dist}')
            # Check early stopping criteria
            if wass_dist < best_wass_dist:
                logging.info(f'Wasserstein distance improved from {best_wass_dist} to {wass_dist}')
                best_wass_dist = wass_dist
                best_gen_weights = generator_model.get_weights()
                best_disc_weights = discriminator_model.get_weights()
                patience_counter = 0
            else:
                logging.info(f'Wasserstein distance did not improve from {best_wass_dist}')
                patience_counter += 1
            
            if patience_counter >= patience:
                logging.info(f"Early stopping on epoch {epoch}. Restoring best weights of epoch {epoch-patience}...")
                generator_model.set_weights(best_gen_weights)  # restore best weights
                discriminator_model.set_weights(best_disc_weights)
                # Save the models
                logging.info('Saving the models...')
                generator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model_{epoch-patience}.h5')
                discriminator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model_{epoch-patience}.h5')
                logging.info('Plotting final generated samples...')

                if args.condition == True:
                    idx = np.random.randint(0, len(dataset_val)-1)
                    batch_size = len(list(dataset_val.as_numpy_iterator())[idx][0])
                    noise = tf.random.normal([batch_size, T_gen*latent_dim, n_features_input])
                    gen_input = [noise, list(dataset_val.as_numpy_iterator())[idx][0]]
                else:
                    noise = tf.random.normal([batch_size, (T_gen+T_condition)*latent_dim, n_features_input])
                    gen_input = noise

                # plot_samples(dataset_train, generator_model, noises, features, T_gen, T_condition, latent_dim, n_features_input, job_id, (epoch-patience), None, args, final=True)
                logging.info('Done')
                break
            else:
                logging.info(f'Early stopping criterion not met. Patience counter:\n\t{patience_counter}')
            
            # Plot the wasserstein distance
            plt.figure(figsize=(10, 6))
            plt.plot(wass_to_plot)
            plt.xlabel('Epoch')
            plt.ylabel('Wasserstein distance')
            plt.title(f'Mean over the features of the Wasserstein distances (Validaiton set)')
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/0_wasserstein_distance.png')
            plt.close()

    # Maybe it is not necessary, but I prefer to clear all the memory
    gc.collect()
    tf.keras.backend.clear_session()
    sys.exit()

