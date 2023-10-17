'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 2 to avoid the messages about the compilation of the CUDA code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_utils import *
from sklearn.preprocessing import StandardScaler
import argparse
import logging
# from tensorflow.keras.utils import plot_model
from model_utils import *
from joblib import dump, load


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-d', '--data', type=str, help='Which data to use (orderbok, message)', default='message')
    parser.add_argument('-N', '--N_days', type=int, help='Number of the day to consider')
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of generator layers', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of discriminator layers', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-c', '--condition', action='store_true', help='Conditioning on the first T_condition time steps')
    parser.add_argument('-tc', '--T_condition', help='Number of time steps to condition on', type=int, default=2)
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
    stock = 'TSLA'
    date = '2015-01-01_2015-01-31_10'

    N = args.N_days
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
    os.mkdir(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.data}_{args.T_condition}_{args.loss}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.data}_{args.T_condition}_{args.loss}') # Generated samples
    os.mkdir(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.data}_{args.T_condition}_{args.loss}') # Models
    
    # Read the message dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()
    message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][:N]

    # Preprocess the data using preprocessing_message_df. This function performs the following operations:
    # 1. Discard the first and last 30 minutes of the day
    # 2. Sample the data at a given frequency
    # 3. Drop the columns that are not needed (Order ID, Time)
    # 4. Filter Event types considering only 1,2,3,4
    res = np.array([preprocessing_message_df(df, discard_time=1800, sampling_freq=40) for df in message_dfs], dtype=object)
    message_dfs, indexes, discarded_time = res[:,0], res[:, 1], res[:, 2]
    logging.info(f'Discarded time (sod, eod):\n\t{discarded_time}')

    # Define the parameters of the GAN. Some of them are set via argparse
    # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
    T_condition = args.T_condition
    window_size = T_condition + 1
    T_real = window_size - T_condition
    n_features_input = message_dfs[0].shape[1]
    latent_dim = 10
    n_epochs = 20000
    batch_size = 32

    # Define the parameters for the early stopping criterion
    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 100

    # Scale all the data at once is prohibitive due to memory resources. For this reason, the data is divided into overlapping pieces
    # and each piece is scaled separately using the partial_fit function of StandardScaler. Note that this is the only scaler that
    # has this feature. Then, the scaled pieces are merged together and divided into windows.
    num_pieces = 5
    for day in range(N):
        logging.info(f'######################### START DAY {day+1}/{N} #########################')

        if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{day+1}.npy'):
            logging.info('\n[Input] ---------- PREPROCESSING ----------')

            data_input = message_dfs[day].values

            # Divide input data into overlapping pieces
            sub_data, length = divide_into_overlapping_pieces(data_input, window_size, num_pieces)

            if sub_data[-1].shape[0] < window_size:     
                raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

            logging.info(f'Number of windows: {length}')

            # Create a scaler object to scale the data
            scaler = StandardScaler()
            logging.info(f'Memorize "sub" estimates of mean and variance...')
            for piece_idx, data in enumerate(sub_data):
                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # The scaler is updated with the data of each piece
                scaler.partial_fit(data)
            logging.info('Done.')

            # save the scaler
            logging.info('Save the scaler...')
            dump(scaler, f'scaler_{day+1}.joblib')
            logging.info('Done.')

            # Create a memmap to store the scaled data.
            final_shape = (length, window_size, n_features_input)
            fp = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape)

            start_idx = 0
            logging.info(f'\nStart scaling the data...')
            for piece_idx, data in enumerate(sub_data):
                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # Here the scaling is performed and the resulting scaled data is assign divided into windows
                # and assigned to the memory mapped vector
                scaled_data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
                windows = np.array(divide_into_windows(scaled_data, window_size))
                logging.info(f'\twindows shape: {windows.shape}')
                end_idx = start_idx + windows.shape[0]
                fp[start_idx:end_idx] = windows
                start_idx = end_idx
                del windows  # Explicit deletion
            logging.info('Done.')

            logging.info('\nSplit the condition data into train and validation sets...')
            train, val = train_test_split(fp, train_size=0.75)
            logging.info(f'Train shape:\n\t{train.shape}\nValidation shape:\n\t{val.shape}')
            logging.info('Done.')

            if args.condition == True:
                logging.info('\nDividing each window into condition and input...')
                condition_train, input_train = train[:, :T_condition, :], train[:, T_condition:, :]
                condition_val, input_val = val[:, :T_condition, :], val[:, T_condition:, :]
                logging.info('Done.')

            logging.info('\nSave the files...')
            np.save(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', condition_train)
            np.save(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', condition_val)
            np.save(f'../data/input_train_{stock}_{window_size}_{day+1}.npy', input_train)
            np.save(f'../data/input_val_{stock}_{window_size}_{day+1}.npy', input_val)
            logging.info('Done.')

            logging.info('\n[Input] ---------- DONE ----------')
        else:
            logging.info('Loading input_train, input_validation and input_test sets...')
            input_train = np.load(f'../data/input_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            input_val = np.load(f'../data/input_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            
            logging.info(f'input_train shape:\n\t{input_train.shape}')
            logging.info(f'condition_train shape:\n\t{condition_train.shape}')
            logging.info(f'input_val shape:\n\t{input_val.shape}')
            logging.info(f'condition_val shape:\n\t{condition_val.shape}')
            
            logging.info('Loading the scaler...')
            scaler = load(f'scaler_{day+1}.joblib')
            logging.info('Done.')

        logging.info(f"\nHYPERPARAMETERS:\n"
                     f"\tgenerator: {args.type_gen}\n"
                     f"\tdiscriminator: {args.type_disc}\n"
                     f"\tn_layers_gen: {args.n_layers_gen}\n"
                     f"\tn_layers_disc: {args.n_layers_disc}\n"
                     f"\tlatent_dim per time: {latent_dim}\n"
                     f"\tn_features: {n_features_input}\n"
                     f"\tn_epochs: {n_epochs}\n"
                     f"\tT_condition: {T_condition}\n"
                     f"\tT_real: {T_real}\n"
                     f"\tbatch_size: {batch_size} (num_batches: {input_train.shape[0]//batch_size})\n"
                     f"\tcondition: {args.condition}\n"
                     f"\tloss: {args.loss}\n"
                     f"\tpatience: {patience}\n"
                     f"\tjob_id: {job_id}\n")

        # Define the optimizers
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        optimizer = [generator_optimizer, discriminator_optimizer]

        # Build the models
        if args.condition== True:
            generator_model = build_generator(args.n_layers_gen, args.type_gen, True, T_real, T_condition, n_features_input, latent_dim, True)
            discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, True, T_real, T_condition, n_features_input, True)
        else:
            generator_model = build_generator(args.n_layers_gen, args.type_gen, True, T_real, T_condition, n_features_input, latent_dim, False)
            discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, True, T_real, T_condition, n_features_input, False)

        logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
        generator_model.summary(print_fn=logging.info)
        logging.info('\n')
        discriminator_model.summary(print_fn=logging.info)
        logging.info('[Model] ---------- DONE ----------\n')

        # Define a dictionary to store the metrics
        metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

        # Define checkpoint and checkpoint manager
        checkpoint_prefix = f"models/{job_id}/"
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                        generator_model=generator_model,
                                        discriminator_model=discriminator_model)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=3)
        # checkpoint.restore(checkpoint_manager.latest_checkpoint)


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

        for epoch in range(n_epochs):
            j = 0
            if args.condition == True:
                for batch_condition, batch_real_samples in dataset_train:
                    j += 1
                    batch_size = batch_real_samples.shape[0]
                    generator_model, discriminator_model = train_step(batch_real_samples, batch_condition, generator_model, discriminator_model, optimizer, args.loss, T_real, T_condition, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, args.condition, args)
            else:
                for batch_real_samples in dataset_train:
                    j += 1
                    batch_condition = np.zeros_like(batch_real_samples)
                    batch_size = batch_real_samples.shape[0]
                    generator_model, discriminator_model = train_step(batch_real_samples, batch_condition, generator_model, discriminator_model, optimizer, args.loss, T_real, T_condition, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, args.condition, args)
            # Save the models via checkpoint
            checkpoint_manager.save()

            logging.info('Creating a time series with the generated samples...')
            number_of_batches_plot = 100 
            features = message_dfs[day].columns
            plot_samples(dataset_train, number_of_batches_plot, generator_model, features, T_real, T_condition, latent_dim, n_features_input, job_id, epoch, scaler, args, final=False)
            logging.info('Done')

            if epoch > 100:
                logging.info('Check Early Stopping Criteria on Validation Set')

                wass_dist = [[] for i in range(n_features_input)]
                if args.condition == True:
                    for val_batch_condition, val_batch in dataset_val:
                        batch_size = val_batch.shape[0]
                        noise = tf.random.normal([batch_size, T_real*latent_dim, n_features_input])
                        generated_samples = generator_model([noise, val_batch_condition], training=True)
                        for feature in range(generated_samples.shape[2]):
                            for i in range(generated_samples.shape[0]):
                                w = wasserstein_distance(val_batch[i, :, feature], generated_samples[i, :, feature])
                                wass_dist[feature].append(w)
                else:
                    for val_batch in dataset_val:
                        batch_size = val_batch.shape[0]
                        noise = tf.random.normal([batch_size, (T_real+T_condition)*latent_dim, n_features_input])
                        generated_samples = generator_model(noise, training=True)
                        for feature in range(generated_samples.shape[2]):
                            for i in range(generated_samples.shape[0]):
                                w = wasserstein_distance(val_batch[i, :, feature], generated_samples[i, :, feature])
                                wass_dist[feature].append(w)

                # Compute the mean of all the wasserstein distances. The idea is to track this quantity
                # that is an aggregate measure of how the generator is able to generate samples that are
                # similar to the real ones.
                wass_dist = np.mean(np.array(wass_dist).flatten())
                logging.info(f'Wasserstein distance: {wass_dist}')
                # Check early stopping criteria
                if wass_dist < best_wass_dist:
                    best_wass_dist = wass_dist
                    best_gen_weights = generator_model.get_weights()
                    best_disc_weights = discriminator_model.get_weights()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logging.info(f"Early stopping on epoch {epoch}. Restoring best weights of epoch {epoch-patience}...")
                    generator_model.set_weights(best_gen_weights)  # restore best weights
                    discriminator_model.set_weights(best_disc_weights)
                    logging.info('Plotting final generated samples...')

                    if args.condition == True:
                        idx = np.random.randint(0, len(dataset_val)-1)
                        batch_size = len(list(dataset_val.as_numpy_iterator())[idx][0])
                        noise = tf.random.normal([batch_size, T_real*latent_dim, n_features_input])
                        gen_input = [noise, list(dataset_val.as_numpy_iterator())[idx][0]]
                    else:
                        noise = tf.random.normal([batch_size, (T_real+T_condition)*latent_dim, n_features_input])
                        gen_input = noise

                    plot_samples(dataset_train, number_of_batches_plot, generator_model, features, T_real, T_condition, latent_dim, n_features_input, job_id, (epoch-patience), scaler, args, final=True)
                    logging.info('Done')
                    break
                else:
                    logging.info(f'Early stopping criterion not met. Patience counter:\n\t{patience_counter}')

        logging.info(f'##################### END DAY {day+1}/{N} #####################\n')