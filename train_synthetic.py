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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Main script used to train the GAN on synthetic data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
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
    parser.add_argument('-sy', '--synthetic', action='store_true', help='Pass it if your data is synthetic.')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    
    # Print the current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    job_id = formatted_datetime

    if not os.path.exists('logs'):
        os.mkdir('logs')
    logging.basicConfig(filename=f'logs/train_{job_id}.log', format='%(message)s', level=levels[args.log])
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    logger = tf.get_logger()
    logger.setLevel('ERROR')

    # Set the seed for TensorFlow to the number of the beast
    tf.random.set_seed(666)

    # Enable device placement logging
    tf.debugging.set_log_device_placement(True)

    # Check the available GPUs
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
   
    # Create a synthetic dataset for testing purposes
    logging.info('\n[Data] ---------- CREATING SYNTHETIC DATASET ----------')
    normal_1 = np.random.normal(-2, 1, (2000))
    normal_2 = np.random.normal(2, 2, (2000))
    data1 = np.concatenate((normal_1, normal_2))
    data2 = np.random.normal(-2, 1, (4000))
    data = {'MixNormal': data1, 'Normal': data2}
    data_df = pd.DataFrame(data)
    logging.info(f'data_df shape:\n\t{data_df.shape}')
    logging.info('[Data] --------------- DONE ---------------\n')

    # Define the parameters of the GAN. Some of them are set via argparse
    T_condition = args.T_condition
    T_gen = args.T_gen
    window_size = T_condition + T_gen
    n_features_input = data_df.shape[1]
    n_features_gen = data_df.shape[1]
    latent_dim = args.latent_dim
    n_epochs = 5000
    batch_size = args.batch_size

    condition_train = np.zeros((data_df.shape[0]//2, T_condition, data_df.shape[1]))
    input_train = np.zeros((data_df.shape[0]//2, T_gen, data_df.shape[1]))

    for i in range(0, condition_train.shape[0]):
        condition_train[i, :, :] = (data_df.iloc[2*i, :].values)**2
        input_train[i, :, :] = data_df.iloc[2*i+1, :].values
    
    # Plot condition and input
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), tight_layout=True)
    axes[0, 0].hist(condition_train[:, 0, 0], bins=50, label='Condition MixNormal')
    axes[0, 1].hist(condition_train[:, 0, 1], bins=50, label='Condition Normal')
    axes[1, 0].hist(input_train[:, 0, 0], bins=50, label='Input MixNormal')
    axes[1, 1].hist(input_train[:, 0, 1], bins=50, label='Input Normal')
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()
    plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/0_condition_input.png')
    
    logging.info(f'input_train shape:\n\t{input_train.shape}')
    logging.info(f'condition_train shape:\n\t{condition_train.shape}')
    

    # Define the parameters for the early stopping criterion
    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 200

    logging.info(f"\nHYPERPARAMETERS:\n"
                    f"\tgenerator: {args.type_gen}\n"
                    f"\tdiscriminator: {args.type_disc}\n"
                    f"\tn_layers_gen: {args.n_layers_gen}\n"
                    f"\tn_layers_disc: {args.n_layers_disc}\n"
                    f"\tskip_connection: {args.skip_connection}\n"
                    f"\tlatent_dim per time: {latent_dim}\n"
                    f"\tn_features_input: {n_features_input}\n"
                    f"\tn_features_gen: {n_features_gen}\n"
                    f"\tfeatures: {data_df.columns}\n"
                    f"\tn_epochs: {n_epochs}\n"
                    f"\tT_condition: {T_condition}\n"
                    f"\tT_gen: {T_gen}\n"
                    f"\tbatch_size: {batch_size} (num_batches: {input_train.shape[0]//batch_size})\n"
                    f"\tloss: {args.loss}\n"
                    f"\tpatience: {patience}\n"
                    f"\tjob_id: {job_id}\n")

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = [generator_optimizer, discriminator_optimizer]

    # Build the models
    generator_model = build_generator(args.n_layers_gen, args.type_gen, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, latent_dim, True)
    discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, True, args.loss)
    feature_extractor = build_feature_extractor(discriminator_model, [i for i in range(1, args.n_layers_disc)])
    exit()

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
    for epoch in range(n_epochs):
        if epoch % 1 == 0:
            logging.info(f'Epoch: {epoch}/{n_epochs}')

        # Create the noise for the generator
        noise_train = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
        j = 0
        start = time.time()
        for batch_condition, batch_real_samples in dataset_train:
            batch_size = batch_real_samples.shape[0]
            noise = noise_train[j*batch_size:(j+1)*batch_size]
            gen_samples_train, real_output, fake_output, discriminator_loss, generator_loss = train_step(batch_real_samples, batch_condition, generator_model, noise, discriminator_model, feature_extractor, optimizer, args.loss, batch_size, j)
            j += 1
        logging.info(f'Epoch {epoch} took {time.time()-start:.2f} seconds.')

        start = time.time()
        # Summarize performance at each epoch
        summarize_performance(real_output, fake_output, discriminator_loss, generator_loss, metrics, job_id, args)
        logging.info(f'Summarizing performance took {time.time()-start:.2f} seconds.')

        if epoch % 25 == 0 and epoch > 0:
            logging.info(f'Plotting the W1 distances at epoch {epoch}...')
            W1_tr = overall_wasserstein_distance(generator_model, dataset_train, noise_train)
            W1_train.append(W1_tr)
            noise_val = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
            W1_v = overall_wasserstein_distance(generator_model, dataset_train, noise_val)
            W1_val.append(W1_v)
            plt.figure(figsize=(10, 6), tight_layout=True)
            plt.plot(W1_train, label='Train')
            plt.plot(W1_val, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Wasserstein distance')
            plt.legend()
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/1_wasserstein_distance.png')
            logging.info('Done.')

        if epoch % 25 == 0 and epoch > 0:
            logging.info(f'Plotting generated samples by the GAN at epoch {epoch}...')
            features = data_df.columns
            plot_samples(dataset_train, generator_model, features, T_gen, n_features_gen, job_id, epoch, args)
            logging.info('Done.')

            logging.info('Saving the models...')
            generator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.keras')
            discriminator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model.keras')
            logging.info('Done')

        if epoch > 2500:
            logging.info('Check Early Stopping Criteria...')
            if W1_train[-1] + 5e-4 < best_wass_dist:
                logging.info(f'Wasserstein distance improved from {best_wass_dist} to {W1_train}')
                best_wass_dist = W1_train[-1]
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
        log_gpu_memory()
        log_memory_usage()
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

