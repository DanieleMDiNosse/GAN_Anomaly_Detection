'''This script contains the training of the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_utils import *
import argparse
import logging
import time
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.utils import plot_model
from model_utils import *
import gc
import sys
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 3 to log only errors of tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf

def ar1_sine(size):
    '''This function generates a variable following an AR(1) process with autoregressive parameter that varies in time.

    Parameters
    ----------
    size : int
        Lenght of the resulting variable.

    Returns
    -------
    X : ndarray of shape (N)
        Synthetic generated variable.
    b : ndarray of shape (N)
        Time varying parameter.

    '''
    np.random.seed(666)
    X = np.zeros(shape=size)
    b = np.zeros(shape=size)
    X[1] = np.random.normal(0, 1)
    for t in range(1, size - 1):
        b[t + 1] = 0.5 * np.sin(np.pi * (t + 1) / int(size / 10))
        X[t + 1] = b[t + 1] * X[t] + np.random.normal(0, 1)
    
    data_df = pd.DataFrame({'X': X, 'b': b})
    return data_df

def sin_wave(amplitude, omega, phi, num_periods, samples_per_period, change_amplitude=False):
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
    data_df = pd.DataFrame({'Sine 1': sine_wave[:time.shape[0]//2], 'Sine 2': sine_wave[time.shape[0]//2:]})
    return data_df

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

def ar1(mu, phi, size):
    time = np.arange(0, size)

    # Generate the AR(1) process
    ar1 = np.zeros(time.shape)
    for t in range(1, len(time)):
        ar1[t] = mu + phi * ar1[t - 1] + np.random.normal(0, 0.5)
    
    data_df = pd.DataFrame({'AR(1)': ar1})
    return data_df

def fit_ARn(time_series, n):
    """
    Fit an AR(n) model to a given time series provided as a numpy array or list.
    
    Parameters:
    - time_series: A list or numpy array containing the time series data.
    
    Returns:
    - A tuple containing the model's fitted parameters and the summary of the model fit.
    """
    # Fit the AR(1) model
    model = AutoReg(time_series, lags=n)
    model_fitted = model.fit()
    
    # Return the fitted parameters and the summary
    return model_fitted.params, model_fitted.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Main script used to train the GAN on synthetic data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-d', '--data', help='Type of synthetic data to generate (sine, ar1_sine, step_fun, ar1)', type=str, default='sine')
    parser.add_argument('-e', '--n_epochs', help='Number of epochs', type=int, default=10000)
    parser.add_argument('-c', '--conditional', action='store_true', help='Use or not conditional GAN. The default is False')
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
    parser.add_argument('-sy', '--synthetic', action='store_true', help='Use synthetic data')

    args = parser.parse_args()
    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    
    # Current date and time
    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    job_id = formatted_datetime

    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Create a log file
    logging.basicConfig(filename=f'logs/train_synthetic_{job_id}.log', format='%(message)s', level=levels[args.log])
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Set the logger of TensorFlow to ERROR
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
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('generated_samples'):
        os.mkdir('generated_samples')
    if not os.path.exists('models'):
        os.mkdir('models')

    os.mkdir(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Generated samples
    os.mkdir(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Models

    # Create a synthetic dataset for testing purposes
    logging.info('\n[Data] ---------- CREATING SYNTHETIC DATASET ----------')
    if args.data == 'sine':
        data_df = sin_wave(1, 0.1, 0, 10, 100, change_amplitude=False)
    if args.data == 'ar1_sine':
        data_df = ar1_sine(1000)
    if args.data == 'step_fun':
        data_df = step_fun(10)
    if args.data == 'ar1':
        mu, phi = 0.40, 0.90
        data_df = ar1(mu, phi, size=1000)

    # Normalize the data
    scaler = StandardScaler()
    data_df = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)
    logging.info(f'data_df shape:\n\t{data_df.shape}')
    logging.info('[Data] --------------- DONE ---------------\n')

    # Define the parameters of the GAN. Some of them are set via argparse
    T_condition = args.T_condition
    T_gen = args.T_gen
    window_size = T_condition + T_gen
    n_features_input = data_df.shape[1]
    n_features_gen = data_df.shape[1]
    latent_dim = args.latent_dim
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Instantiate two empty vectors to store the condition and the input to the GAN
    condition_train = np.zeros((data_df.shape[0]//2, T_condition, data_df.shape[1]))
    input_train = np.zeros((data_df.shape[0]//2, T_gen, data_df.shape[1]))

    for i in range(0, condition_train.shape[0]):
        condition_train[i, :, :] = data_df.iloc[2*i, :].values
        input_train[i, :, :] = data_df.iloc[2*i+1, :].values
    
    # Plot condition and input
    fig, axes = plt.subplots(2, n_features_input, figsize=(10, 6), tight_layout=True)
    if n_features_input == 1:
        axes[0].hist(condition_train[:, :, 0], bins=50, label=f'Condition {data_df.columns[0]}', alpha=0.8)
        axes[1].hist(input_train[:, :, 0], bins=50, label=f'Input {data_df.columns[0]}', alpha=0.8)
        axes[0].legend()
        axes[1].legend()
    else:
        for i in range(n_features_input):
            axes[0, i].hist(condition_train[:, :, i], bins=50, label=f'Condition {data_df.columns[i]}', alpha=0.8)
            axes[1, i].hist(input_train[:, :, i], bins=50, label=f'Input {data_df.columns[i]}', alpha=0.8)
            axes[0, i].legend()
            axes[1, i].legend()
    plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/0_condition_input.png')
    
    logging.info(f'input_train shape:\n\t{input_train.shape}')
    logging.info(f'condition_train shape:\n\t{condition_train.shape}')
    
    # Define the parameters for the early stopping criterion
    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 500

    logging.info(f"\nHYPERPARAMETERS:\n"
                    f"\tgenerator: {args.type_gen}\n"
                    f"\tdiscriminator: {args.type_disc}\n"
                    f"\tn_layers_gen: {args.n_layers_gen}\n"
                    f"\tn_layers_disc: {args.n_layers_disc}\n"
                    f"\tskip_connection: {args.skip_connection}\n"
                    f"\tconditional: {args.conditional}\n"
                    f"\tlatent_dim per time: {latent_dim}\n"
                    f"\tn_features_input: {n_features_input}\n"
                    f"\tn_features_gen: {n_features_gen}\n"
                    f"\tfeatures: {data_df.columns}\n"
                    f"\tn_epochs: {n_epochs}\n"
                    f"\tT_condition: {T_condition}\n"
                    f"\tT_gen: {T_gen}\n"
                    f"\tbatch_size: {batch_size} (num_batches: {input_train.shape[0]//batch_size})\n"
                    f"\tloss: {args.loss}\n"
                    f"\tclipping disc gradients: {args.clipping}\n"
                    f"\tpatience: {patience}\n"
                    f"\tjob_id: {job_id}\n")

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = [generator_optimizer, discriminator_optimizer]

    # Build the models
    generator_model = build_generator(args.n_layers_gen, args.type_gen, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, latent_dim, args.conditional)
    discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, args.skip_connection, T_gen, T_condition, n_features_input, n_features_gen, args.conditional, args.loss)
    if args.type_disc == 'dense':
        if args.skip_connection:
            if args.conditional:
                layers_to_extract = [7+8*i for i in range(args.n_layers_disc)]
            else:
                layers_to_extract = [4+4*i for i in range(args.n_layers_disc)]
        else:
            if args.conditional:
                layers_to_extract = [7+6*i for i in range(args.n_layers_disc)]
            else:
                layers_to_extract = [4+3*i for i in range(args.n_layers_disc)]
        layers_to_extract.append(len(discriminator_model.layers)-2)
    if args.type_disc == 'lstm':
        layers_to_extract = [4+4*i for i in range(args.n_layers_disc)]
        layers_to_extract.append(len(discriminator_model.layers)-2)
    if args.type_disc == 'conv':
        layers_to_extract = [4+4*i for i in range(args.n_layers_disc)]
        layers_to_extract.append(len(discriminator_model.layers)-2)
    feature_extractor = build_feature_extractor(discriminator_model, [i for i in layers_to_extract])

    logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
    generator_model.summary(print_fn=logging.info)
    logging.info('\n')
    discriminator_model.summary(print_fn=logging.info)
    logging.info('[Model] ---------- DONE ----------\n')

    # Define a dictionary to store the metrics
    metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

    # Train the GAN.
    logging.info('\n[Training] ---------- START TRAINING ----------')
    if args.conditional == True:
        dataset_train = tf.data.Dataset.from_tensor_slices((condition_train, input_train)).batch(batch_size)
    else:
        dataset_train = tf.data.Dataset.from_tensor_slices(input_train).batch(batch_size)
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
    # for batch in dataset_train:
    #     logging.info(f'{batch}')
    # exit()

    num_batches = len(dataset_train)
    logging.info(f'Number of batches:\n\t{num_batches}\n')

    W1_train = []
    W1_val = []
    delta_monitor = 25
    for epoch in range(n_epochs):
        if epoch % delta_monitor == 0:
            logging.info(f'Epoch: {epoch}/{n_epochs}')
            start = time.time()

        # Create the noise for the generator
        noise_train = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
        j = 0
        for batch in dataset_train:
            if args.conditional == True:
                batch_condition, batch_real_samples = batch
            else:
                batch_real_samples = batch
            batch_size = batch_real_samples.shape[0]
            noise = noise_train[j*batch_size:(j+1)*batch_size]
            if args.conditional:
                gen_samples_train, real_output, fake_output, discriminator_loss, generator_loss = train_step(batch_real_samples, batch_condition, generator_model, noise, discriminator_model, feature_extractor, optimizer, args.loss, batch_size, args.clipping)
            else:
                gen_samples_train, real_output, fake_output, discriminator_loss, generator_loss = train_step_unconditional(batch_real_samples, generator_model, noise, discriminator_model, feature_extractor, optimizer, args.loss, batch_size, args.clipping)
            j += 1
        if epoch % delta_monitor == 0: logging.info(f'Epoch {epoch} took {time.time()-start:.2f} seconds.')

        if epoch % delta_monitor//2 == 0 and epoch > 0:
            summarize_performance(real_output, fake_output, discriminator_loss, generator_loss, metrics, job_id, args)

        if epoch % delta_monitor == 0 and epoch > 0:
            logging.info(f'Plotting the W1 distances at epoch {epoch}...')
            W1_tr, gen_samples_train = overall_wasserstein_distance(generator_model, dataset_train, noise_train, args.conditional)
            W1_train.append(W1_tr)
            noise_val = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
            W1_v, gen_samples_val = overall_wasserstein_distance(generator_model, dataset_train, noise_val, args.conditional)
            W1_val.append(W1_v)
            plt.figure(figsize=(10, 6), tight_layout=True)
            plt.plot(W1_train, label='Train')
            plt.plot(W1_val, label='Validation')
            plt.xlabel('Epoch')
            plt.ylabel('Wasserstein distance')
            plt.legend()
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/1_wasserstein_distance.png')
            logging.info('Done.')

        if epoch % (delta_monitor*5) == 0 and epoch > 0:
            logging.info(f'Plotting generated samples by the GAN at epoch {epoch}...')
            features = data_df.columns
            plot_samples(dataset_train, generator_model, features, T_gen, n_features_gen, job_id, epoch, args)
            logging.info('Done.')

            if args.data == 'ar1':
                logging.info('Fitting and AR(1) model to the generated samples (back-transformed)...')
                logging.info(f'Real parameters:\n\tmu: {mu}\n\tphi: {phi}')
                # inverse transform the generated samples
                gen_samples_train_inv = scaler.inverse_transform(gen_samples_train)
                for i in range(n_features_gen):
                    params, summary = fit_ARn(gen_samples_train_inv[:, i], 1)
                    logging.info(f'Generated parameters: \n\tmu: {params[0]:.2f}\n\tphi: {params[1]:.2f}\n')
                logging.info('Done.')
            logging.info('Saving the models...')
            generator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/generator_model.keras', save_format='h5')
            discriminator_model.save(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/discriminator_model.keras', save_format='h5')
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
    
    noise_val = tf.random.normal([input_train.shape[0], T_gen*latent_dim, n_features_gen])
    generated_samples_train = []
    generated_samples_val = []
    real_samples = []
    j = 0
    for batch_condition, batch in dataset_train:
        noise_t = noise_train[j*batch_size:(j+1)*batch_size]
        noise_v = noise_val[j*batch_size:(j+1)*batch_size]
        gen_sample_train = generator_model([noise_t, batch_condition])
        gen_sample_val = generator_model([noise_v, batch_condition])
        for i in range(gen_sample_train.shape[0]):
            # All the appended samples will be of shape (T_gen, n_features_gen)
            generated_samples_train.append(gen_sample_train[i, -1, :])
            generated_samples_val.append(gen_sample_val[i, -1, :])
            real_samples.append(batch[i, -1, :])
        j += 1
    plot_pca_with_marginals(generated_samples_train, real_samples, f'{job_id}_train', args)
    plot_pca_with_marginals(generated_samples_val, real_samples, f'{job_id}_val', args)
    logging.info('Done.')


    logging.info('Computing the errors on the correlation matrix using bootstrap...')
    # At the end of the training, compute the errors on the correlation matrix using bootstrap.
    # In order to do so, I need the best generator and the noises used.
    correlation_matrix(dataset_train, generator_model, noise, T_gen, n_features_gen, job_id)
    logging.info('Done.')

    # Maybe it is not necessary, but I prefer to clear all the memory and exit the script
    gc.collect()
    tf.keras.backend.clear_session()
    sys.exit()


'''TODO
- Compute correlation between Y_real(t) and Y_gen(t+k) for k=1,..,N. Thi will give me
an idea of whether or not the generator takes the data and just shift it. '''