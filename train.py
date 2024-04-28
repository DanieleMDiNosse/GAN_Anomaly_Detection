'''This script contains the training of the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_utils import *
import argparse
import logging
import time
# from tensorflow.keras.utils import plot_model
from model_utils import *
import gc
import sys
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 3 to log only errors of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Main script used to train the GAN.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-s', '--stock', type=str, help='Stock to consider')
    parser.add_argument('-dt', '--date', type=str, help='Date in the folder name')
    parser.add_argument('-N', '--N_days', type=int, help='Number of the day to consider')
    parser.add_argument('-pd', '--previous_days', action='store_true', help='If passed, it will consider days until the Nth one')
    parser.add_argument('-d', '--depth', help='Depth of the orderbook', type=int)
    parser.add_argument('-e', '--n_epochs', help='Number of epochs', type=int, default=100000)
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
    parser.add_argument('-cl', '--clipping', action='store_true', help='Use or not weight clipping')
    parser.add_argument('-lo', '--load', help='Load a model. The job_id must be provided', type=int, default=0)
    parser.add_argument('-sy', '--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('-c', '--conditional', action='store_true', help='Use conditional GAN')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    
    # Print the current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    if os.getenv("PBS_JOBID") != None:
        job_id = f'{os.getenv("PBS_JOBID")}_{formatted_datetime}'
    else:
        job_id = f'{os.getpid()}_{formatted_datetime}'

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(filename=f'logs/train_{job_id}.log', format='%(message)s', level=levels[args.log])
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    logger = tf.get_logger()
    logger.setLevel('ERROR')

    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # Set the seed for TensorFlow to the number of the beast
    tf.random.set_seed(666)

    # Enable device placement logging
    tf.debugging.set_log_device_placement(True)
   
    # Define data parameters
    stock = args.stock
    date = args.date
    N = args.N_days
    depth = args.depth

    logging.info(f'Stock:\n\t{stock}')
    if args.previous_days: 
        logging.info(f'Number of days:\n\t{N}')
    else:
        logging.info(f'Day number:\n\t{N}')

    # Check the available GPUs
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        print("No GPUs available.")
    else:
        print("Available GPUs:")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        for device in physical_devices:
            print(f'\t{device}\n')
    
    # Folders creation
    os.mkdir(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Generated samples
    os.mkdir(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Models
   
    dataframes_folder_path = f'../data/{stock}_{date}'
    # Check if the extensions of the files are .csv or .parquet
    paths = os.listdir(dataframes_folder_path)
    for path in paths:
        if path.endswith('.csv'):
            logging.info('\n[Data] ---------- PREPROCESSING ----------')
            logging.info('Converting the orderbook csv files into parquet dataframes...')
            convert_and_renamecols(f'{dataframes_folder_path}/{path}')
            logging.info('[Data] -------------- DONE --------------\n')

    # Create the orderbook dataframe
    logging.info('\n[Data] ---------- CREATING ORDERBOOK SNAPSHOTS ----------')
    orderbook_df, prices_change = create_LOB_snapshots(stock, date, N, depth, previous_days=args.previous_days)
    logging.info(f'Orderbook input dataframe shape:\n\t{orderbook_df.shape}')
    logging.info('[Data] --------------- DONE ---------------\n')

    # Define the parameters of the GAN. Some of them are set via argparse
    T_condition = args.T_condition
    T_gen = args.T_gen
    window_size = T_condition + T_gen
    n_features_input = orderbook_df.shape[1]
    n_features_gen = 2*depth
    latent_dim = args.latent_dim
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Define the parameters for the early stopping criterion
    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 1000

    if not os.path.exists(f'../data/{stock}_{date}/miscellaneous/input_train_{stock}_{window_size}_{N}days_orderbook_.npy'):
        logging.info('\n[Input&Condition] ---------- CREATING INPUT AND CONDITION ----------')
        logging.info('Dividing each window (T_gen+T_condition) into condition and input...')
        condition_train = np.zeros((orderbook_df.shape[0]//2, T_condition, n_features_input))
        input_train = np.zeros((orderbook_df.shape[0]//2, T_gen, n_features_input))

        for i in range(0, condition_train.shape[0]):
            condition_train[i, :, :] = orderbook_df.iloc[2*i, :].values
            input_train[i, :, :] = orderbook_df.iloc[2*i+1, :].values

        logging.info(f'input_train shape:\n\t{input_train.shape}')
        logging.info(f'condition_train shape:\n\t{condition_train.shape}')
        logging.info('Done.')

        logging.info('\nSave the condition_input and input_train .npy files...')
        np.save(f'../data/{stock}_{date}/miscellaneous/condition_train_{stock}_{window_size}_{N}days_orderbook.npy', condition_train)
        np.save(f'../data/{stock}_{date}/miscellaneous/input_train_{stock}_{window_size}_{N}days_orderbook.npy', input_train)
        logging.info('Done.')
        logging.info('[Input&Condtion] --------------- DONE ---------------\n')

    elif os.path.exists(f'../data/{stock}_{date}/miscellaneous/input_train_{stock}_{window_size}_{N}days_orderbook.npy'):
        logging.info('Loading input_train and condition_train...')
        input_train = np.load(f'../data/{stock}_{date}/miscellaneous/input_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
        condition_train = np.load(f'../data/{stock}_{date}/miscellaneous/condition_train_{stock}_{window_size}_{N}days_orderbook.npy', mmap_mode='r')
        
        logging.info(f'input_train shape:\n\t{input_train.shape}')
        logging.info(f'condition_train shape:\n\t{condition_train.shape}')
        logging.info('Done.')

    # Create bar plots showing the empirical distribution of the LOB snapshots at time t and t+1
    logging.info('Creating bar plots showing the empirical distribution of the LOB snapshots at time t and t+T_gen...')
    bar_plot_levels(stock, date, N, window_size, c=50)
    logging.info('Done.')

    logging.info(f"\nHYPERPARAMETERS:\n"
                    f"\tstock: {stock}\n"
                    f"\tdepth: {depth}\n"
                    f"\tgenerator: {args.type_gen}\n"
                    f"\tdiscriminator: {args.type_disc}\n"
                    f"\tn_layers_gen: {args.n_layers_gen}\n"
                    f"\tn_layers_disc: {args.n_layers_disc}\n"
                    f"\tskip_connection: {args.skip_connection}\n"
                    f"\tconditional: {args.conditional}\n"
                    f"\tlatent_dim per time: {latent_dim}\n"
                    f"\tbatch_size: {batch_size} (num_batches: {input_train.shape[0]//batch_size})\n"
                    f"\tT_condition: {T_condition}\n"
                    f"\tT_gen: {T_gen}\n"
                    f"\tloss: {args.loss}\n"
                    f"\tpatience: {patience}\n"
                    f"\tn_features_input: {n_features_input}\n"
                    f"\tn_features_gen: {n_features_gen}\n"
                    f"\tfeatures: {orderbook_df.columns}\n"
                    f"\tn_epochs: {n_epochs}\n"
                    f"\tjob_id: {job_id}\n"
                    f"\tLoaded model: {None if args.load==0 else args.load}\n")

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = [generator_optimizer, discriminator_optimizer]

    logging.info('\n[Model] ---------- BUILDING MODELS ----------')
    if args.load == 0:
        # Build the models
        logging.info('Building the generator model...')
        generator_model = build_generator(args.n_layers_gen, args.type_gen, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, latent_dim, True)
        logging.info('Building the discriminator model...')
        discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, True, args.loss)
        logging.info('Building the feature extractor...')
        if args.type_disc == 'dense':
            if args.skip_connection:
                layers_to_extract = [7+8*i for i in range(args.n_layers_disc)]
            else:
                layers_to_extract = [7+6*i for i in range(args.n_layers_disc)]
            layers_to_extract.append(len(discriminator_model.layers)-2)
        if args.type_disc == 'lstm':
            layers_to_extract = [4+4*i for i in range(args.n_layers_disc)]
            layers_to_extract.append(len(discriminator_model.layers)-2)
        if args.type_disc == 'conv':
            layers_to_extract = [4+4*i for i in range(args.n_layers_disc)]
            layers_to_extract.append(len(discriminator_model.layers)-2)
        feature_extractor = build_feature_extractor(discriminator_model, [i for i in layers_to_extract])
    else:
        prev_job_id = args.load
        # Load the models
        logging.info(f'Loading the models from job_id {prev_job_id}...')
        generator_model = tf.keras.models.load_model(f'models/{prev_job_id}.pbs01_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.h5')
        discriminator_model = tf.keras.models.load_model(f'models/{prev_job_id}.pbs01_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model.h5')
        feature_extractor = build_feature_extractor(discriminator_model, [i for i in range(1, args.n_layers_disc)])
        logging.info('Done.')
    logging.info('[Model] --------------- DONE ---------------\n')


    logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
    generator_model.summary(print_fn=logging.info)
    logging.info('\n')
    discriminator_model.summary(print_fn=logging.info)
    logging.info('[Model] ---------- DONE ----------\n')

    # Define a dictionary to store the metrics
    metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

    # Train the GAN.
    logging.info('\n[Training] ---------- START TRAINING ----------')
    dataset_train = tf.data.Dataset.from_tensor_slices((condition_train, input_train)).batch(batch_size)
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

    num_batches = len(dataset_train)
    logging.info(f'Number of batches:\n\t{num_batches}\n')

    W1_train = []
    W1_val = []
    delta_monitor = 25
    for epoch in range(n_epochs):
        if epoch % 10 == 0:
            logging.info(f'Epoch: {epoch}/{n_epochs}')
            start = time.time()

        # Create the noise for the generator
        noise_train = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
        j = 0
        for batch_condition, batch_real_samples in dataset_train:
            batch_size = batch_real_samples.shape[0]
            noise = noise_train[j*batch_size:(j+1)*batch_size]
            gen_samples_train, real_output, fake_output, discriminator_loss, generator_loss = train_step(batch_real_samples, batch_condition, generator_model, noise, discriminator_model, feature_extractor, optimizer, args.loss, batch_size, j)
            j += 1
        if epoch % 10 == 0: logging.info(f'Epoch {epoch} took {time.time()-start:.2f} seconds.')

        if epoch == 3: exit()
        # Summarize performance at each epoch
        if epoch % delta_monitor == 0 and epoch > 0:
            summarize_performance(real_output, fake_output, discriminator_loss, generator_loss, metrics, job_id, args)

        if epoch % delta_monitor == 0 and epoch > 0:
            logging.info(f'Plotting the W1 distances at epoch {epoch}...')
            W1_tr, gen_samples_train = overall_wasserstein_distance(generator_model, dataset_train, noise_train, conditional=args.conditional)
            W1_train.append(W1_tr)
            noise_val = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
            W1_v, gen_samples_val  = overall_wasserstein_distance(generator_model, dataset_train, noise_val, conditional=args.conditional)
            W1_val.append(W1_v)
            plt.figure(figsize=(10, 6), tight_layout=True)
            plt.plot(W1_train, label='Train')
            plt.plot(W1_val, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Wasserstein distance')
            plt.legend()
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/1_wasserstein_distance.png')
            logging.info('Done.')

        if epoch % (delta_monitor*10) == 0 and epoch > 0:
            logging.info(f'Plotting generated samples by the GAN at epoch {epoch}...')
            features = orderbook_df.columns
            plot_samples(dataset_train, generator_model, features, T_gen, n_features_gen, job_id, epoch, args)
            logging.info('Done.')

            logging.info('Saving the models...')
            generator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.keras')
            discriminator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model.keras')
            logging.info('Done')

        if epoch > 50000:
            logging.info('Check Early Stopping Criteria...')
            if W1_val[-1] + 5e-4 < best_wass_dist:
                logging.info(f'Wasserstein distance improved from {best_wass_dist} to {W1_train}')
                best_wass_dist = W1_val[-1]
                best_gen_weights = generator_model.get_weights()
                best_disc_weights = discriminator_model.get_weights()
                patience_counter = 0
                np.save(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/noise_{epoch}.npy', noise)
            else:
                logging.info(f'Wasserstein distance did not improve from {best_wass_dist}')
                patience_counter += 1
            
            if patience_counter >= patience:
                best_epoch = epoch - patience
                logging.info(f"Early stopping on epoch {epoch}. Restoring best weights of epoch {best_epoch}...")
                generator_model.set_weights(best_gen_weights)  # restore best weights
                discriminator_model.set_weights(best_disc_weights)

                logging.info('Saving the models...')
                generator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.h5')
                discriminator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model.h5')
                logging.info('Done')
            else:
                logging.info(f'Early stopping criterion not met. Patience counter:\n\t{patience_counter}')

        # Memory management
        # log_gpu_memory()
        # log_memory_usage()
        free_memory()

        if patience_counter >= patience:
            break
    
    logging.info('[Training] ---------- DONE ----------\n')

    logging.info('Plotting the first 2 principal components of the generated and real samples...')
    # Plot the first 2 principal components of the generated and real samples
    # Load the best generator
    generator_model = tf.keras.models.load_model(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.h5')
    
    generated_samples = []
    real_samples = []
    k = 0
    for batch_condition, batch in dataset_train:
        gen_sample = generator_model([noises[k], batch_condition])
        for i in range(gen_sample.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples.append(gen_sample[i, -1, :])
            real_samples.append(batch[i, -1, :])
        k += 1
    plot_pca_with_marginals(generated_samples, real_samples, job_id, args)
    logging.info('Done.')


    logging.info('Computing the errors on the correlation matrix using bootstrap...')
    # At the end of the training, compute the errors on the correlation matrix using bootstrap.
    # In order to do so, I need the best generator and the noises used.
    correlation_matrix(dataset_train, generator_model, noises, T_gen, n_features_gen, job_id)
    logging.info('Done.')

    # Maybe it is not necessary, but I prefer to clear all the memory and exit the script
    gc.collect()
    tf.keras.backend.clear_session()
    sys.exit()

