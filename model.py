'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import *
import argparse
import logging
# from tensorflow.keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import Input, Conv2D, Dense, Concatenate, Reshape, Dropout, LeakyReLU, Conv2DTranspose, BatchNormalization, Flatten, GaussianNoise, LSTM
from tensorflow.keras.models import Model

def build_conditioner(T_cond, num_features_condition, cond_units):
    '''Build the conditioner model. The conditioner takes as input the condition and outputs a vector of K values.
    The condition is processed by a LSTM layer and the output of the LSTM layer is processed by a dense layer.
    The conditioner is trained to extract the relevant information from the condition.
    
    Parameters
    ----------
    T_cond : int
        Length of the condition.
    num_features : int
        Number of features of the condition.
    cond_units : list
        List of the number of hidden units for the LSTM layers.
    
    Returns
    -------
    conditioner_model : tensorflow.keras.Model
        The conditioner model.'''

    K, hidden_units0 = cond_units[0], cond_units[1]

    condition_input = Input(shape=(T_cond, num_features_condition), name="condition_input")
    x = LSTM(hidden_units0, return_sequences=True, name="1_lstm")(condition_input)
    x = LSTM(hidden_units0, return_sequences=False, name="2_lstm")(x)
    x = Dense(K, name="condition_output")(x)
    output_cond = LeakyReLU(0.2)(x)

    conditioner_model = Model(condition_input, output_cond, name="condition_model")
    return conditioner_model

def build_generator(latent_dim, gen_units, T_real, num_features_input, activate_condition):
    '''Build the generator model. The generator takes as input the condition and the noise and outputs a sample.
    The condition is processed by a LSTM layer and the noise is processed by a LSTM layer. Then the two outputs are concatenated
    and processed by a dense layer. The output of the dense layer is reshaped to have the same shape of the real samples.
    The generator is trained to fool the discriminator.
    
    Parameters
    ----------
    T_cond : int
        Length of the condition.
    latent_dim : int
        Dimension of the noise.
    gen_units : list
        List of the number of hidden units for the LSTM layers.
    T_real : int
        Length of the real samples.
    num_features : int
        Number of features of the real samples.
    
    Returns
    -------
    generator_model : tensorflow.keras.Model
        The generator model.
    condition_model : tensorflow.keras.Model
        The condition model.
    '''

    K = gen_units[0]
    # conv_units = gen_units[2]
    # kernel_size = gen_units[3]

    if not activate_condition:
        input = Input(shape=(latent_dim*T_real), name='noise_input')
        zeros = tf.zeros((tf.shape(input)[0], K))
        x = Concatenate(axis=-1, name='concatenation')([zeros, input])
    else:
        noise_input = Input(shape=(latent_dim*T_real), name='noise_input')
        condition_output = Input(shape=(K,), name='condition_input_from_conditioner')
        x = Concatenate(axis=-1, name='concatenation')([condition_output, noise_input])

    x = Dense(T_real*num_features_input*5, name='1_dense')(x)
    x = BatchNormalization(name='1_batch_norm')(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(T_real*num_features_input*10, name='2_dense')(x)
    x = BatchNormalization(name='2_batch_norm')(x)
    x = LeakyReLU(0.2)(x)

    x = Dense(T_real*num_features_input*8, name='3_dense')(x)
    x = BatchNormalization(name='3_batch_norm')(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((T_real,num_features_input,8), name='1_reshape')(x)

    x = Conv2D(filters=4, kernel_size=(5,1), strides=(1,1), padding="same", name='1_conv2d')(x)
    x = BatchNormalization(name='4_batch_norm')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(filters=1, kernel_size=(5,1), strides=(1,1), padding="same", name='2_conv2d')(x)
    x = BatchNormalization(name='5_batch_norm')(x)
    x = LeakyReLU(0.2)(x)

    output = Reshape((T_real,num_features_input), name='2_reshape')(x)

    if not activate_condition:
        generator_model = Model(input, output, name="generator_model") 
    else:
        generator_model = Model([condition_output, noise_input], output, name="generator_model")
    return generator_model

def build_discriminator(T_real, num_features_input, disc_units, activate_condition):
    '''Build the discriminator model. The discriminator takes as input the condition and the sample and outputs a real value.
    In a WGAN the discriminator does not classify samples, but it rather outputs a real value evaluating their realism, thus we refer to it as a discriminator.
    The condition is taken as the output of the condition model and then it is processed in order to match the real sample dimensions.
    Together with the real sample, the condition (reshaped) is concatenated and processed by a LSTM layer. The output of the LSTM layer
    is processed by a dense layer and then by a sigmoid layer. The discriminator is trained to distinguish real samples from fake samples.
    
    Parameters
    ----------
    T_real : int
        Length of the real samples.
    T_cond : int
        Length of the condition.
    num_features : int
        Number of features of the real samples.
    disc_units : list
        List of the number of hidden units for the LSTM layers.
    
    Returns
    -------
    discriminator_model : tensorflow.keras.Model
        The discriminator model.'''

    K = disc_units[0]
    # Input for the condition
    input_real = Input(shape=(T_real, num_features_input), name='input')
    x = Reshape((T_real, num_features_input, 1), name='1_reshape')(input_real) # (None, T_real, num_features_input, 1)
    
    if activate_condition:
        condition_input = Input(shape=(K,), name='condition_input_from_conditioner')

    # x = Conv2D(filters=32, kernel_size=(5,1), strides=(2,1), padding="same", name='1_conv2D')(x) # (None, 125, 5, 32)
    x = Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding="same", name='1_conv2D')(x) # (None, 125, 5, 32)
    x = GaussianNoise(0.2)(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=64, kernel_size=(5,5), strides=(5,1), padding="same", name='2_conv2D')(x) # (None, 25, 10, 64)
    # x = BatchNormalization()(x)
    x = GaussianNoise(0.2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x) # (None, 10240)
    x = Dense(K)(x)
    x = LeakyReLU(0.2)(x)

    if not activate_condition:
        zeros = tf.zeros((tf.shape(x)[0], K))
        x = Concatenate(axis=-1, name='concatenation')([zeros, x])
        output = Dense(1, name='dense_output', activation='sigmoid')(x)
        discriminator_model = Model(input_real, output, name='discriminator_model')
    else:
        x = Concatenate(axis=-1, name='concatenation')((x, condition_input))
        output = Dense(1, name='dense_output', activation='sigmoid')(x) # Goal: 0 for fake samples, 1 for real samples
        discriminator_model = Model([condition_input, input_real], output, name='discriminator_model')
    # plot_model(discriminator_model, to_file=f'plots/{job_id}/discriminator_model_plot.png', show_shapes=True, show_layer_names=True)

    return discriminator_model

# @tf.function
def train_step(real_samples, conditions, condition_model, generator_model, discriminator_model, optimizer, batch_size, latent_dim, j, epoch, metrics, threshold_condition, activate_condition):
    '''Train the GAN for one batch.
    
    Parameters
    ----------
    real_samples : numpy.ndarray
        The real samples.
    conditions : numpy.ndarray
        The conditions.
    condition_model : tensorflow.keras.Model
        The condition model.
    generator_model : tensorflow.keras.Model
        The generator model.
    discriminator_model : tensorflow.keras.Model
        The discriminator model.
    T_cond : int
        Length of the condition.
    latent_dim : int
        Dimension of the noise.
    i : int
        Batch number.
    metrics : dict
        Dictionary to store the metrics.
    
    Returns
    -------
    condition_model : tensorflow.keras.Model
        The condition model.
    generator_model : tensorflow.keras.Model
        The generator model.
    discriminator_model : tensorflow.keras.Model
        The discriminator model.'''

    discriminator_optimizer, generator_optimizer = optimizer

    # Create a GradientTape for the conditioner, generator, and discriminator.
    # GrandietTape collects all the operations that are executed inside it.
    # Then this operations are used to compute the gradients of the loss function with 
    # respect to the trainable variables. Remember that 'persistent=True' is needed
    # iff you want to compute the gradients of the operations inside the tape more than once
    # (for example wrt different losses)

    # Discriminator training
    for _ in range(1):
        # Initialize random noise
        noise = tf.random.normal([batch_size, latent_dim*T_real])
        with tf.GradientTape(persistent=True) as tape:
            # Ensure the tape is watching the trainable variables
            tape.watch(discriminator_model.trainable_variables)
            tape.watch(condition_model.trainable_variables)
            if activate_condition:
                # Step 1: Use condition_model to preprocess conditions and get K values
                k_values = condition_model(conditions, training=True)
                # Step 2: Generate fake samples using generator
                generated_samples = generator_model([k_values, noise], training=True)
                # Step 3: discriminator distinguishes real and fake samples
                real_output = discriminator_model([k_values, real_samples], training=True)
                fake_output = discriminator_model([k_values, generated_samples], training=True)
            else:
                # Step 1: Generate fake samples using generator
                generated_samples = generator_model(noise, training=True)
                # Add some noise to the real and generated samples
                real_samples_noised = real_samples + tf.random.normal(real_samples.shape, mean=0.0, stddev=0.5)
                generated_samples_noised = generated_samples + tf.random.normal(generated_samples.shape, mean=0.0, stddev=0.5)
                # Step 2: discriminator distinguishes real and fake samples
                real_output = discriminator_model(real_samples_noised, training=True)
                fake_output = discriminator_model(generated_samples_noised, training=True)
            # Step 4: Compute the losses
            discriminator_loss = compute_discriminator_loss(real_output, fake_output)

        # Calculate gradients
        gradients_of_discriminator = tape.gradient(discriminator_loss, discriminator_model.trainable_variables)
        # Apply gradients to update weights
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

        if activate_condition:
            # Calculate gradients
            gradients_of_conditioner = tape.gradient(discriminator_loss, condition_model.trainable_variables)
            # Apply gradients to update weights
            discriminator_optimizer.apply_gradients(zip(gradients_of_conditioner, condition_model.trainable_variables))
    
    # Delete the tape to free resources
    del tape

    # Generator training
    for _ in range(1):
        noise = tf.random.normal([batch_size, latent_dim*T_real])
        with tf.GradientTape(persistent=True) as tape:
            # Ensure the tape is watching the trainable variables
            tape.watch(condition_model.trainable_variables)
            tape.watch(generator_model.trainable_variables)
            if activate_condition:
                # Step 1: Use condition_model to preprocess conditions and get K values
                k_values = condition_model(conditions, training=True)
                # Step 2: Generate fake samples using generator
                generated_samples = generator_model([k_values, noise], training=True)
                # Step 3: discriminator distinguishes real and fake samples
                fake_output = discriminator_model([k_values, generated_samples], training=True)
            else:
                # Step 1: Generate fake samples using generator
                generated_samples = generator_model(noise, training=True)
                # Step 2: discriminator distinguishes real and fake samples
                fake_output = discriminator_model(generated_samples, training=True)
            # Compute the losses
            gen_loss = compute_generator_loss(fake_output)

    # Calculate gradients
    gradients_of_generator = tape.gradient(gen_loss, generator_model.trainable_variables)
    # Apply gradients to update weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    if activate_condition:
        # Calculate gradients
        gradients_of_conditioner = tape.gradient(gen_loss, condition_model.trainable_variables)
        # Apply gradients to update weights
        generator_optimizer.apply_gradients(zip(gradients_of_conditioner, condition_model.trainable_variables))

    # Delete the tape to free resources
    del tape

    if j % 100 == 0:
        if activate_condition:
            logging.info(f'ON | Epoch: {epoch} | Batch: {j}/{num_batches} | Disc loss: {np.mean(discriminator_loss):.5f} | Gen loss: {np.mean(gen_loss):.5f} | <Disc_output_r>: {real_output.numpy()[1,:].mean():.5f}| <Disc_output_f>: {fake_output.numpy()[1,:].mean():.5f}')
        else:
            logging.info(f'OFF | Epoch: {epoch} | Batch: {j}/{num_batches} | Disc loss: {np.mean(discriminator_loss):.5f} | Gen loss: {np.mean(gen_loss):.5f} | <Disc_output_r>: {real_output.numpy().mean():.5f}| <Disc_output_f>: {fake_output.numpy().mean():.5f}')

        models = [generator_model, discriminator_model]
        gradients = [gradients_of_generator, gradients_of_discriminator]
        models_name = ['GENERATOR', 'DISCRIMINATOR']

        for model, gradients_of_model, name in zip(models, gradients, models_name):
            logging.info(f'\t{name}:')
            for grad, var in zip(gradients_of_model, model.trainable_variables):
                grad_norm = tf.norm(grad).numpy()
                logging.info(f"\tLayer {var.name}, Gradient Norm: {grad_norm:.5f}")

    if j % 300 == 0:
        summarize_performance(real_output, fake_output, discriminator_loss, gen_loss, generated_samples, real_samples, metrics, j, epoch, threshold_condition)
        condition_model.save(f'models/{job_id}/condition_model.h5')
        generator_model.save(f'models/{job_id}/generator_model.h5')
        discriminator_model.save(f'models/{job_id}/discriminator_model.h5')
    return condition_model, generator_model, discriminator_model

def compute_discriminator_loss(real_output, fake_output):
    '''Compute the discriminator loss.
    
    Parameters
    ----------
    real_output : numpy.ndarray
        The output of the discriminator for the real samples.
    fake_output : numpy.ndarray
        The output of the discriminator for the fake samples.
    
    Returns
    -------
    total_disc_loss : float
        The discriminator loss.'''

    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss

def compute_generator_loss(fake_output):
    '''Compute the generator loss.
    
    Parameters
    ----------
    fake_output : numpy.ndarray
        The output of the discriminator for the fake samples.
    
    Returns
    -------
    float
        The generator loss.'''
    return -tf.math.log(fake_output + 1e-10)
    # return tf.reduce_mean(binary_crossentropy(tf.ones_like(fake_output), fake_output))

def summarize_performance(real_output, fake_output, discriminator_loss, gen_loss, generated_samples, real_samples, metrics, i, epoch, threshold_condition):
    '''Summarize the performance of the GAN.
    
    Parameters
    ----------
    real_output : numpy.ndarray
        The output of the discriminator for the real samples.
    fake_output : numpy.ndarray
        The output of the discriminator for the fake samples.
    discriminator_loss : float
        The discriminator loss.
    gen_loss : float
        The generator loss.
    generated_samples : numpy.ndarray
        The generated samples.
    metrics : dict
        Dictionary to store the metrics.
    
    Returns
    -------
    None'''

    generated_samples = generated_samples[0,:,:].numpy()
    real_samples = real_samples[0,:,:].numpy()
    features = ['Event type', 'Size', 'Price', 'Direction']

    # add the metrics to the dictionary
    metrics['discriminator_loss'].append(np.mean(discriminator_loss))
    metrics['gen_loss'].append(np.mean(gen_loss))
    for score in real_output[1,:]:
        metrics['real_disc_out'].append(score)
    for score in fake_output[1,:]:
        metrics['fake_disc_out'].append(score)

    y_max = max(np.max(metrics['discriminator_loss']), np.max(metrics['gen_loss']).max())
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['discriminator_loss'], label='Discriminator loss')
    plt.plot(metrics['gen_loss'], label='Generator loss')
    plt.xlabel('Batch')
    plt.xticks(np.arange(300, len(metrics['discriminator_loss'])+1, step=300), labels=np.arange(300, len(metrics['real_disc_out'])+1, step=300).astype('str'))
    plt.ylabel('Loss')
    plt.vlines(threshold_condition/300, 0, y_max, linestyles='dashed', alpha=0.6)
    plt.legend()
    plt.savefig(f'plots/{job_id}/losses.png')

    y_max = max(np.max(metrics['real_disc_out']), np.max(metrics['fake_disc_out']))
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_disc_out'], label='Real')
    plt.plot(metrics['fake_disc_out'], label='Fake')
    plt.xlabel('Batch')
    plt.xticks(np.arange(300, len(metrics['real_disc_out'])+1, step=300), labels=np.arange(300, len(metrics['real_disc_out'])+1, step=300).astype('str'))
    plt.ylabel('Discriminator output')
    plt.vlines(threshold_condition/300, 0, y_max, linestyles='dashed', alpha=0.6)
    plt.legend()
    plt.savefig(f'plots/{job_id}/disc_output.png')

    # Plot a chosen generated sample
    fig, axes = plt.subplots(generated_samples.shape[1], 1, figsize=(10, 10), tight_layout=True)
    for j, feature in zip(range(generated_samples.shape[1]), features):
        axes[j].plot(generated_samples[:, j], label=f'Generated {feature}')
        axes[j].set_title(f'Generated {feature}')
    axes[j].set_xlabel('Time (Events)')
    plt.savefig(f'plots/{job_id}/generated_samples_{epoch}_{i}.png')

    # Plot a chosen real sample
    fig, axes = plt.subplots(real_samples.shape[1], 1, figsize=(10, 10), tight_layout=True)
    for j, feature in zip(range(real_samples.shape[1]), features):
        axes[j].plot(real_samples[:, j])
        axes[j].set_title(f'Real {feature}')
    axes[j].set_xlabel('Time (Events)')
    plt.savefig(f'plots/{job_id}/real_samples.png')

    plt.close('all')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('stock', type=str, help='Which stock to use (TSLA, MSFT)')
    parser.add_argument('-N', '--N_days', type=int, help='Number of the day to consider')

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

    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Enable device placement logging
    tf.debugging.set_log_device_placement(True)
   
    # Load the data
    stock = args.stock
    if stock == 'TSLA':
        date = '2015-01-01_2015-01-31_10'
    elif stock == 'MSFT':
        date = '2018-04-01_2018-04-30_5'

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
    os.mkdir(f'plots/{job_id}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}') # Generated samples
    os.mkdir(f'models/{job_id}') # Models
    
    # Read the message dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    message_df_paths = [path for path in dataframes_paths if 'message' in path]
    message_df_paths.sort()
    message_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in message_df_paths][:N]

    # Preprocess the data using preprocessing_message_df. This function performs the following operations:
    # 1. Drop the columns that are not needed (Order ID)
    # 2. Filter Event types considering only 1,2,3,4
    # 3. Filter the data considering only the elements from the 1000th (due to the volatility estimation)
    res = np.array([preprocessing_message_df(df) for df in message_dfs], dtype=object)
    message_dfs, indexes = res[:,0], res[:, 1]

    # Define the parameters of the GAN.
    # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
    window_size = 500
    n_features_condition = message_dfs[0].shape[1]
    n_features_input = n_features_condition - 1
    latent_dim = 15
    n_epochs = 100
    T_condition = int(window_size*0.75)
    T_real = window_size - T_condition
    n_units_generator = 100
    batch_size = 64
    # output condition, hidden units condition, filter dim, kernel size
    gen_units = [20, 64, 5, 25]
    cond_units = [20, 64, 5, 25]
    disc_units = [gen_units[0], 64, 5, 25]

    num_pieces = 10
    for day in range(N):
        logging.info(f'######################### START DAY {day+1}/{N} #########################')

        if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{day+1}.npy'):
            logging.info('\n[Input] ---------- PREPROCESSING ----------')

            data_input = message_dfs[day].values
            # Divide input data into overlapping pieces
            sub_data = divide_into_overlapping_pieces(data_input, window_size, num_pieces)
            num_windows = 0
            for data in sub_data:
                num_windows += data.shape[0]-window_size+1
            logging.info(f'Number of windows: {num_windows}')

            logging.info(f'Dividing data into windows')
            num_windows = 0
            for piece_idx, data in enumerate(sub_data):

                if sub_data[-1].shape[0] < window_size:
                    raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

                # Each piece is divided into windows
                windows = np.array(divide_into_windows(data, window_size))
                num_windows += windows.shape[0]
                del windows  # Explicit deletion
            logging.info(f'Number of windows: {num_windows}')

            # Create a memmap to store the data. The first shape is the number of windows (samples) for each piece
            # multiplied by the number of pieces.
            final_shape_input = (num_windows, window_size, n_features_input+1)
            fp_input = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape_input)

            start_idx = 0
            for piece_idx, data in enumerate(sub_data):
                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                windows = np.array(divide_into_windows(data, window_size))
                end_idx = start_idx + windows.shape[0]
                fp_input[start_idx:end_idx] = windows
                start_idx = end_idx
                del windows  # Explicit deletion
            logging.info('Done.')

            logging.info('\nSplit the condition data into train, validation and test sets...')
            train, test = fp_input[:int(fp_input.shape[0]*0.75)], fp_input[int(fp_input.shape[0]*0.75):]
            train, val = train[:int(train.shape[0]*0.75)], train[int(train.shape[0]*0.75):]
            logging.info('Done.')

            logging.info('\nDividing each window into condition and input...')
            condition_train, input_train = train[:, :T_condition, :], train[:, T_condition:, 1:]
            condition_val, input_val = val[:, :T_condition, :], val[:, T_condition:, 1:]
            condition_test, input_test = test[:, :T_condition, :], test[:, T_condition:, 1:]
            logging.info('Done.')

            logging.info('\nSave the files...')
            np.save(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', condition_train)
            np.save(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', condition_val)
            np.save(f'../data/condition_test_{stock}_{window_size}_{day+1}.npy', condition_test)
            np.save(f'../data/input_train_{stock}_{window_size}_{day+1}.npy', input_train)
            np.save(f'../data/input_val_{stock}_{window_size}_{day+1}.npy', input_val)
            np.save(f'../data/input_test_{stock}_{window_size}_{day+1}.npy', input_test)
            logging.info('Done.')

            logging.info('\n[Input] ---------- DONE ----------')
        else:
            logging.info('Loading input_train, input_validation and input_test sets...')
            input_train = np.load(f'../data/input_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            input_val = np.load(f'../data/input_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            input_test = np.load(f'../data/input_test_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_test = np.load(f'../data/condition_test_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            
            logging.info('Done.')

        # Use logging.info to print all the hyperparameters
        logging.info(f'\nHYPERPARAMETERS:\n\tlatent_dim: {latent_dim}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_real: {T_real}\n\tFeatures Input: {message_dfs[day].columns}\n\tbatch_size: {batch_size}\n\tnum_batches: {input_train.shape[0]//batch_size}')

        # Define the optimizers
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # Define the loss function
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        optimizer = [generator_optimizer, discriminator_optimizer]

        condition_model = build_conditioner(T_condition, n_features_condition, cond_units)
        generator_model = build_generator(latent_dim, gen_units, T_real, n_features_input, activate_condition=False)
        discriminator_model = build_discriminator(T_real, n_features_input, disc_units, activate_condition=False)

        logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
        condition_model.summary(print_fn=logging.info)
        logging.info('\n')
        generator_model.summary(print_fn=logging.info)
        logging.info('\n')
        discriminator_model.summary(print_fn=logging.info)
        logging.info('[Model] ---------- DONE ----------\n')

        # Define a dictionary to store the metrics
        metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

        # Train the GAN. When I load the data, I use the argument mmap_mode='r' to avoid to load the data in memory.
        # This is because the data is too big to fit in memory. This means that the data is loaded in memory only when
        # when it is needed.
        num_subvectors = 5
        slice = int(input_train.shape[0] / num_subvectors)
        for i in range(num_subvectors):
            logging.info(f'\n---------- TRAINING ON PIECE {i} ----------')
            input_train = input_train[i*slice: (i+1)*slice]
            condition_train = condition_train[i*slice: (i+1)*slice]
            logging.info(f'input_train shape:\n\t{input_train.shape}')
            logging.info(f'condition_train shape:\n\t{condition_train.shape}')
            dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
            global num_batches
            num_batches = len(dataset)
            logging.info(f'Number of batches:\n\t{num_batches}\n')
            if i > 0:
                # Load the models of the previous training (previous piece)
                logging.info('Loading models from the previous piece...')
                condition_model = tf.keras.models.load_model(f'models/{job_id}/condition_model.h5')
                generator_model = tf.keras.models.load_model(f'models/{job_id}/generator_model.h5')
                discriminator_model = tf.keras.models.load_model(f'models/{job_id}/discriminator_model.h5')
            for epoch in range(n_epochs):
                j = 0
                for batch_real_samples, batch_conditions in dataset:
                    j += 1
                    batch_size = batch_real_samples.shape[0]
                    # threshold_condition = int(num_batches/5)
                    threshold_condition = num_batches*3
                    # if j < threshold_condition:
                    if i < 3:
                        condition_model, generator_model, discriminator_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, discriminator_model, optimizer, batch_size, latent_dim, j, epoch, metrics, threshold_condition, activate_condition=False)
                    else:
                        condition_model, generator_model, discriminator_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, discriminator_model, optimizer, batch_size, latent_dim, j, epoch, metrics, threshold_condition, activate_condition=True)
            
                # Create another time the models and load the weights of the trained one. This is done due to the fact
                # that in the first part of the training the condition is not active and the input shapes of the models
                # are different.
                weights = [model.get_weights() for model in [condition_model, generator_model, discriminator_model]]

                condition_model = build_conditioner(T_condition, n_features_condition, cond_units)
                generator_model = build_generator(latent_dim, gen_units, T_real, n_features_input, activate_condition=False)
                discriminator_model = build_discriminator(T_real, n_features_input, disc_units, activate_condition=False)

                condition_model.set_weights(weights[0])
                generator_model.set_weights(weights[1])
                discriminator_model.set_weights(weights[2])

            # save the models
            condition_model.save(f'models/{job_id}/condition_model.h5')
            generator_model.save(f'models/{job_id}/generator_model.h5')
            discriminator_model.save(f'models/{job_id}/discriminator_model.h5')

        # Remeber to handle the last piece
        logging.info(f'---------- Training on the LAST piece ----------')
        input_train = input_train[(i+1)*slice:]
        condition_train = condition_train[(i+1)*slice:]
        dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
        global last_num_batches
        last_num_batches = len(dataset)
        condition_model = tf.keras.models.load_model(f'models/{job_id}/condition_model.h5')
        generator_model = tf.keras.models.load_model(f'models/{job_id}/generator_model.h5')
        discriminator_model = tf.keras.models.load_model(f'models/{job_id}/discriminator_model.h5')
        for epoch in range(n_epochs):
            j = 0
            for batch_real_samples, batch_conditions in dataset:
                batch_size = batch_real_samples.shape[0]
                threshold_condition = int(last_num_batches/5)
                if j < threshold_condition:
                    condition_model, generator_model, discriminator_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, discriminator_model, optimizer, batch_size, latent_dim, j, epoch, metrics, threshold_condition, activate_condition=False)
                elif j == threshold_condition:
                    weights = [model.get_weights() for model in [condition_model, generator_model, discriminator_model]]
                    condition_model = build_conditioner(T_condition, n_features_condition, cond_units)
                    generator_model = build_generator(latent_dim, gen_units, T_real, n_features_input, activate_condition=True)
                    discriminator_model = build_discriminator(T_real, n_features_input, disc_units, activate_condition=True)
                    condition_model.set_weights(weights[0])
                    generator_model.set_weights(weights[1])
                    discriminator_model.set_weights(weights[2])
                    condition_model, generator_model, discriminator_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, discriminator_model, optimizer, batch_size, latent_dim, j, epoch, metrics, activate_condition=True)
                else:
                    condition_model, generator_model, discriminator_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, discriminator_model, optimizer, batch_size, latent_dim, j, epoch, metrics, activate_condition=True)
            # Create another time the models and load the weights of the trained one. This is done due to the fact
            # that in the first part of the training the condition is not active and the input shapes of the models
            # are different.
            weights = [model.get_weights() for model in [condition_model, generator_model, discriminator_model]]
            condition_model = build_conditioner(T_condition, n_features_condition, cond_units)
            generator_model = build_generator(latent_dim, gen_units, T_real, n_features_input, activate_condition=False)
            discriminator_model = build_discriminator(T_real, n_features_input, disc_units, activate_condition=False)
            condition_model.set_weights(weights[0])
            generator_model.set_weights(weights[1])
            discriminator_model.set_weights(weights[2])
        
        # save the models
        condition_model.save(f'models/{job_id}/condition_model.h5')
        generator_model.save(f'models/{job_id}/generator_model.h5')
        discriminator_model.save(f'models/{job_id}/discriminator_model.h5')

        logging.info(f'##################### END DAY {day+1}/{N} #####################\n')