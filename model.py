'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import *
from sklearn.preprocessing import StandardScaler
import argparse
import logging
# from tensorflow.keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.keras.layers import Input, Conv2D, Dense, Concatenate, Reshape, Dropout, LeakyReLU, MaxPooling1D, UpSampling1D, BatchNormalization, Flatten, GaussianNoise, LSTM, ReLU, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import backend

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

def build_generator(T_real, num_features_input):

    input = Input(shape=(T_real, num_features_input), name='input')
    
    # Encoding Path
    encode_1 = LSTM(units=128, return_sequences=True)(input)
    downsample_1 = MaxPooling1D(pool_size=2)(encode_1)

    encode_2 = LSTM(units=64, return_sequences=True)(downsample_1)
    downsample_2 = MaxPooling1D(pool_size=2)(encode_2)

    encode_3 = LSTM(units=32, return_sequences=True)(downsample_2)
    downsample_3 = MaxPooling1D(pool_size=2)(encode_3)

    encode_4 = LSTM(units=16, return_sequences=True)(downsample_3)
    downsample_4 = MaxPooling1D(pool_size=2)(encode_4)

    # Bottleneck
    bottleneck = LSTM(units=8, return_sequences=True)(downsample_4)

    # Decoding Path
    upsample_1 = UpSampling1D(size=2)(bottleneck)
    conc = Concatenate()([upsample_1, encode_4])
    decode_1 = LSTM(units=16, return_sequences=True)(conc)

    upsample_2 = UpSampling1D(size=2)(decode_1)
    conc = Concatenate()([upsample_2, encode_3])
    decode_2 = LSTM(units=32, return_sequences=True)(conc)

    upsample_3 = UpSampling1D(size=2)(decode_2)
    conc = Concatenate()([upsample_3, encode_2])
    decode_3 = LSTM(units=64, return_sequences=True)(conc)

    upsample_4 = UpSampling1D(size=2)(decode_3)
    conc = Concatenate()([upsample_4, encode_1])
    decode_4 = LSTM(units=128, return_sequences=True)(conc)

    output = TimeDistributed(Dense(units=num_features_input))(decode_4)
    # output = LSTM(units=num_features_input, return_sequences=True)(decode_4)

    generator_model = Model([input], output, name="generator_model")

    #Plot the model
    # plot_model(generator_model, to_file='generator_model_plot.png', show_shapes=True, show_layer_names=True)

    return generator_model

def build_discriminator(T_real, num_features_input):

    input = Input(shape=(T_real, num_features_input), name='input')
    x = LSTM(units=32, return_sequences=True)(input)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=128, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    output = Dense(1, name='dense_output', activation='sigmoid')(x) # Goal: 0 for fake samples, 1 for real samples
    discriminator_model = Model([input], output, name='discriminator_model')

    #Plot the model
    # plot_model(discriminator_model, to_file='discriminator_model_plot.png', show_shapes=True, show_layer_names=True)

    return discriminator_model

def train_step_test(real_samples, generator_model, discriminator_model, optimizer, latent_dim, batch_size, num_batches, j, epoch, metrics):
    discriminator_optimizer, generator_optimizer = optimizer

    noise = tf.random.normal([batch_size, latent_dim, real_samples.shape[2]])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(discriminator_model.trainable_variables)
        generated_samples = generator_model(noise, training=True)
        real_output = discriminator_model(real_samples, training=True)
        fake_output = discriminator_model(generated_samples, training=True)
        disc_loss = compute_discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = tape.gradient(disc_loss, discriminator_model.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    
    del tape

    noise = tf.random.normal([batch_size, latent_dim, real_samples.shape[2]])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(generator_model.trainable_variables)
        generated_samples = generator_model(noise, training=True)
        fake_output = discriminator_model(generated_samples, training=True)
        gen_loss = compute_generator_loss(fake_output)

        gradients_of_generator = tape.gradient(gen_loss, generator_model.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    del tape

    disc_accuracy = compute_accuracy([real_output, fake_output])
    if j % 10 == 0:
        logging.info(f'{str} | Epoch: {epoch} | Batch: {j}/{num_batches} | Disc loss: {np.mean(disc_loss):.5f} | Gen loss: {np.mean(gen_loss):.5f} | <Disc_output_r>: {np.mean(real_output):.5f}| <Disc_output_f>: {np.mean(fake_output):.5f} | Disc accuracy: {disc_accuracy:.3f}')
    if j % 50 == 0:
        summarize_performance(real_output, fake_output, disc_loss, gen_loss, generated_samples, real_samples, metrics, j, num_batches, epoch, checkpoint_manager)

    models = [generator_model, discriminator_model]
    gradients = [gradients_of_generator, gradients_of_discriminator]
    models_name = ['GENERATOR', 'DISCRIMINATOR']

    for model, gradients_of_model, name in zip(models, gradients, models_name):
        logging.info(f'\t{name}:')
        for grad, var in zip(gradients_of_model, model.trainable_variables):
            grad_norm = tf.norm(grad).numpy()
            logging.info(f"\tLayer {var.name}, Gradient Norm: {grad_norm:.5f}")

    return generator_model, discriminator_model

# @tf.function
def train_step(real_samples, conditions, generator_model, discriminator_model, optimizer, batch_size, num_batches, j, epoch, metrics):

    generator_optimizer, discriminator_optimizer = optimizer

    # Create a GradientTape for the conditioner, generator, and discriminator.
    # GrandietTape collects all the operations that are executed inside it.
    # Then this operations are used to compute the gradients of the loss function with 
    # respect to the trainable variables. Remember that 'persistent=True' is needed
    # iff you want to compute the gradients of the operations inside the tape more than once
    # (for example wrt different losses)

    # Discriminator training
    for _ in range(1):
        with tf.GradientTape(persistent=True) as tape:
            # Ensure the tape is watching the trainable variables
            tape.watch(discriminator_model.trainable_variables)
            # Step 1: Generate fake samples using generator
            generated_samples = generator_model(conditions, training=True)
            # Step 2: discriminator distinguishes real and fake samples
            real_output = discriminator_model(real_samples, training=True)
            fake_output = discriminator_model(generated_samples, training=True)
            # Step 4: Compute the losses
            discriminator_loss = compute_discriminator_loss(real_output, fake_output)

        # Calculate gradients
        gradients_of_discriminator = tape.gradient(discriminator_loss, discriminator_model.trainable_variables)
        # Apply gradients to update weights
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    
    # Delete the tape to free resources
    del tape

    # Generator training
    for _ in range(1):
        with tf.GradientTape(persistent=True) as tape:
            # Ensure the tape is watching the trainable variables
            tape.watch(generator_model.trainable_variables)
            # Step 1: Generate fake samples using generator
            generated_samples = generator_model(conditions, training=True)
            # Step 2: discriminator distinguishes real and fake samples
            fake_output = discriminator_model(generated_samples, training=True)
            # Compute the losses
            gen_loss = compute_generator_loss(fake_output)

    # Calculate gradients
    gradients_of_generator = tape.gradient(gen_loss, generator_model.trainable_variables)
    # Apply gradients to update weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    # Delete the tape to free resources
    del tape

    if j % 100 == 0:
        logging.info(f'Epoch: {epoch} | Batch: {j}/{num_batches} | Disc loss: {np.mean(discriminator_loss):.5f} | Gen loss: {np.mean(gen_loss):.5f} | <Disc_output_r>: {real_output.numpy().mean():.5f}| <Disc_output_f>: {fake_output.numpy().mean():.5f}')

        models = [generator_model, discriminator_model]
        gradients = [gradients_of_generator, gradients_of_discriminator]
        models_name = ['GENERATOR', 'DISCRIMINATOR']

        for model, gradients_of_model, name in zip(models, gradients, models_name):
            logging.info(f'\t{name}:')
            for grad, var in zip(gradients_of_model, model.trainable_variables):
                grad_norm = tf.norm(grad).numpy()
                logging.info(f"\tLayer {var.name}, Gradient Norm: {grad_norm:.5f}")

    if j % 300 == 0:
        summarize_performance(real_output, fake_output, discriminator_loss, gen_loss, generated_samples, real_samples, metrics, j, epoch)

    return generator_model, discriminator_model

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
    return -tf.math.log(fake_output + 1e-5)
    # return binary_crossentropy(tf.ones_like(fake_output), fake_output)

def wasserstein_loss(predictions):
    '''Compute the Wasserstein loss via an approximation of the Kantorovich-Rubenstein formula.
    
    Parameters
    ----------
    predictions : list
        List of the predictions of the critic for the real&fake/fake samples.
    
    Returns
    -------
    w_tot : float
        The Wasserstein loss.'''
    
    if len(predictions) == 2:
        real_output, fake_output = predictions
        w_real = backend.mean(real_output)
        w_fake = backend.mean(fake_output)
        w_tot = w_fake - w_real
    else:
        fake_output = predictions
        w_tot = -backend.mean(fake_output)
    return w_tot

def gradient_penalty(critic_model, batch_size, real_samples, fake_samples, k_values):
    # Get the interpolated samples
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
    diff = fake_samples - real_samples
    interpolated = real_samples + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic_model([k_values, interpolated], training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return float(gp)

def summarize_performance(real_output, fake_output, discriminator_loss, gen_loss, generated_samples, real_samples, metrics, i, num_batches, epoch, checkpoint_manager):
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

    checkpoint_manager.save()

    # add the metrics to the dictionary
    metrics['discriminator_loss'].append(np.mean(discriminator_loss))
    metrics['gen_loss'].append(np.mean(gen_loss))
    for score in real_output[1,:]:
        metrics['real_disc_out'].append(score)
    for score in fake_output[1,:]:
        metrics['fake_disc_out'].append(score)

    # y_max = max(np.max(metrics['discriminator_loss']), np.max(metrics['gen_loss']).max())
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['discriminator_loss'], label='Discriminator loss')
    plt.plot(metrics['gen_loss'], label='Generator loss')
    plt.xlabel('Batch')
    # plt.xticks(np.arange(300, len(metrics['discriminator_loss'])+1, step=300), labels=np.arange(300, len(metrics['real_disc_out'])+1, step=300).astype('str'))
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{job_id}/losses.png')

    # y_max = max(np.max(metrics['real_disc_out']), np.max(metrics['fake_disc_out']))
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_disc_out'], label='Real')
    plt.plot(metrics['fake_disc_out'], label='Fake')
    plt.xlabel('Batch')
    # plt.xticks(np.arange(300, len(metrics['real_disc_out'])+1, step=300), labels=np.arange(300, len(metrics['real_disc_out'])+1, step=300).astype('str'))
    plt.ylabel('Discriminator output')
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

    # Create an animated gif of the generated samples
    if (i+50)/num_batches > 1:
        logging.info('Creating animated gif...')
        create_animated_gif(job_id, epoch, args.type_gen, args.type_disc, args.n_layers, args.data)
        logging.info('Done.')

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
    # 1. Drop the columns that are not needed (Order ID, Time)
    # 2. Filter Event types considering only 1,2,3,4
    # 3. Filter the data discarding start and end of day
    res = np.array([preprocessing_message_df(df) for df in message_dfs], dtype=object)
    message_dfs, indexes = res[:,0], res[:, 1]

    # Define the parameters of the GAN.
    # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
    window_size = 512
    n_features_input = message_dfs[0].shape[1]
    latent_dim = 100
    n_epochs = 100
    T_condition = int(window_size*0.5)
    T_real = window_size - T_condition
    batch_size = 2

    num_pieces = 5
    for day in range(N):
        logging.info(f'######################### START DAY {day+1}/{N} #########################')

        if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{day+1}_.npy'):
            logging.info('\n[Input] ---------- PREPROCESSING ----------')

            data_input = message_dfs[day].values

            # Divide input data into pieces
            sub_data = divide_into_overlapping_pieces(data_input, window_size, num_pieces)

            if sub_data[-1].shape[0] < window_size:     
                raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

            num_windows = data_input.shape[0] - window_size
            logging.info(f'Number of windows: {num_windows}')

            # Create a scaler object to scale the condition data
            scaler = StandardScaler()
            logging.info(f'Memorize "sub" estimates of mean and variance...')
            for piece_idx, data in enumerate(sub_data):
                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # The scaler is updated with the data of each piece
                scaler.partial_fit(data)
            logging.info('Done.')

            # Create a memmap to store the scaled data.
            final_shape = (num_windows, window_size, n_features_input)
            fp = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape)

            start_idx = 0
            logging.info(f'\nStart scaling the data...')
            for piece_idx, data in enumerate(sub_data):
                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # Here the scaling is performed and the resulting scaled data is assign divided into windows
                # and assigned to the memmap
                scaled_data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
                windows = np.array(divide_into_windows(scaled_data, window_size))
                logging.info(f'\twindows shape: {windows.shape}')
                end_idx = start_idx + windows.shape[0]
                fp[start_idx:end_idx] = windows
                start_idx = end_idx
                del windows  # Explicit deletion
            logging.info('Done.')

            logging.info('\nSplit the condition data into train, validation and test sets...')
            train, test = train_test_split(fp, train_size=0.75)
            train, val = train_test_split(train, train_size=0.75)
            logging.info('Done.')

            logging.info('\nDividing each window into condition and input...')
            condition_train, input_train = train[:, :T_condition, :], train[:, T_condition:, :]
            condition_val, input_val = val[:, :T_condition, :], val[:, T_condition:, :]
            condition_test, input_test = test[:, :T_condition, :], test[:, T_condition:, :]
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
            
            logging.info(f'input_train shape:\n\t{input_train.shape}')
            logging.info(f'condition_train shape:\n\t{condition_train.shape}')
            logging.info(f'input_val shape:\n\t{input_val.shape}')
            logging.info(f'condition_val shape:\n\t{condition_val.shape}')
            logging.info(f'input_test shape:\n\t{input_test.shape}')
            logging.info(f'condition_test shape:\n\t{condition_test.shape}')

            logging.info('Done.')

        # Use logging.info to print all the hyperparameters
        logging.info(f'\nHYPERPARAMETERS:\n\tlatent_dim: {latent_dim}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_real: {T_real}\n\tFeatures Input: {message_dfs[day].columns}\n\tbatch_size: {batch_size}\n\tnum_batches: {input_train.shape[0]//batch_size}')

        # Define the optimizers
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        optimizer = [generator_optimizer, discriminator_optimizer]

        # Define the loss function
        binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # Define the models
        generator_model = build_generator(T_real, n_features_input)
        discriminator_model = build_discriminator(T_real, n_features_input)

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
        dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
        num_batches = len(dataset)
        logging.info(f'Number of batches:\n\t{num_batches}\n')

        for epoch in range(n_epochs):
            j = 0
            for batch_real_samples, batch_conditions in dataset:
                j += 1
                batch_size = batch_real_samples.shape[0]
                generator_model, discriminator_model = train_step(batch_real_samples, batch_conditions, generator_model, discriminator_model, optimizer, batch_size, num_batches, j, epoch, metrics, checkpoint_manager)
            # Save the models via checkpoint
            checkpoint_manager.save()

        logging.info(f'##################### END DAY {day+1}/{N} #####################\n')