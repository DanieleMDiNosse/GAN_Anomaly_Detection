'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import *
from joblib import dump, load
import argparse
import logging
# from tensorflow.keras.utils import plot_model
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Conv1D, Conv2D, Dense, Concatenate, Reshape, TimeDistributed, Dropout, LeakyReLU, Conv2DTranspose, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend


def build_generator(T_cond, latent_dim, gen_units, T_real, num_features_condition, num_features_input):
    '''Build the generator model. The generator takes as input the condition and the noise and outputs a sample.
    The condition is processed by a LSTM layer and the noise is processed by a LSTM layer. Then the two outputs are concatenated
    and processed by a dense layer. The output of the dense layer is reshaped to have the same shape of the real samples.
    The generator is trained to fool the critic.
    
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
    # ----------------- CONDITIONER -----------------
    K, hidden_units0 = gen_units[0], gen_units[1]
    condition_input = Input(shape=(T_cond, num_features_condition, 1))
    x = Conv2D(filters=32, kernel_size=(5,num_features_condition), strides=(1,1), padding="same", activation=LeakyReLU(0.2))(condition_input)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(5,num_features_condition), strides=(5,1), padding="same", activation=LeakyReLU(0.2))(x)
    x = Flatten()(x)
    x = Dense(K)(x)
    output_cond = LeakyReLU(0.2)(x)
    # output = TimeDistributed(dense_layer)(lstm)
    condition_model = Model(condition_input, output_cond, name="condition_model")
    # plot_model(condition_model, to_file=f'plots/{job_id}/condition_model_plot.png', show_shapes=True, show_layer_names=True)

    # ----------------- GENERATOR -----------------
    conv_units = gen_units[2]
    kernel_size = gen_units[3]

    condition_output = Input(shape=(K,))
    print(condition_output.shape)
    noise_input = Input(shape=(latent_dim,))
    print(noise_input.shape)
    x = Concatenate(axis=-1)([condition_output, noise_input]) # (None, 100 + 5)
    x = Dense(25*1*256)(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    x = Reshape((25,1,256))(x)

    x = Conv2DTranspose(filters=128, kernel_size=(5,1), strides=(2,5), padding="same", activation=LeakyReLU(0.2))(x) # (None, 50, 5, 128)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(filters=64, kernel_size=(5,1), strides=(5,1), padding="same", activation=LeakyReLU(0.2))(x) # (None, 250, 5, 64)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(filters=1, kernel_size=(5,1), strides=(1,1), padding="same", activation='tanh')(x) # (None, 250, 5, 1)
    output = Reshape((250,5))(x) # (None, 250, 5)

    generator_model = Model([condition_output, noise_input], output, name="generator_model")
    # plot_model(generator_model, to_file=f'plots/{job_id}/generator_model_plot.png', show_shapes=True, show_layer_names=True)
    return generator_model, condition_model

def build_critic(T_real, T_cond, num_features_input, disc_units):
    '''Build the critic model. The critic takes as input the condition and the sample and outputs a real value.
    In a WGAN the discriminator does not classify samples, but it rather outputs a real value evaluating their realism, thus we refer to it as a Critic.
    The condition is taken as the output of the condition model and then it is processed in order to match the real sample dimensions.
    Together with the real sample, the condition (reshaped) is concatenated and processed by a LSTM layer. The output of the LSTM layer
    is processed by a dense layer and then by a sigmoid layer. The critic is trained to distinguish real samples from fake samples.
    
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
    critic_model : tensorflow.keras.Model
        The critic model.'''

    # Input for the condition
    K, hidden_units0 = disc_units[0], disc_units[1]
    condition_input = Input(shape=(K,))
    x = Dense(T_real*num_features_input)(condition_input)
    x = LeakyReLU(0.2)(x)
    input_cond = Reshape((T_real, num_features_input))(x)

    # Input for the real samples
    conv_units1 = disc_units[2]
    kernel_size = disc_units[3]
    input_real = Input(shape=(T_real, num_features_input)) # (None, T_real, num_features_input)
    x = Concatenate(axis=-1)([input_cond, input_real]) # (None, T_real, num_features_input + num_features_input)
    x = Reshape((T_real, num_features_input*2, 1))(x) # (None, T_real, num_features_input*2, 1)

    x = Conv2D(filters=32, kernel_size=(5,1), strides=(2,1), padding="same", activation=LeakyReLU(0.2))(x) # (None, 125, 10, 32)
    x = Conv2D(filters=64, kernel_size=(5,1), strides=(5,2), padding="same", activation=LeakyReLU(0.2))(x) # (None, 25, 5, 64)
    x = Conv2D(filters=128, kernel_size=(5,1), strides=(5,1), padding="same", activation=LeakyReLU(0.2))(x) # (None, 5, 5, 128)

    x = Flatten()(x) # (None, 10240)
    output = Dense(1, activation='linear')(x) # Goal: high value for real samples, low value for fake samples

    critic_model = Model([condition_input, input_real], output, name='critic_model')
    # plot_model(critic_model, to_file=f'plots/{job_id}/critic_model_plot.png', show_shapes=True, show_layer_names=True)

    return critic_model

# @tf.function
def train_step(real_samples, conditions, condition_model, generator_model, critic_model, optimizer, batch_size, T_cond, latent_dim, j, epoch, metrics):
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
    critic_model : tensorflow.keras.Model
        The critic model.
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
    critic_model : tensorflow.keras.Model
        The critic model.'''

    critic_optimizer, generator_optimizer, conditioner_optimizer = optimizer

    # Create a GradientTape for the conditioner, generator, and critic.
    # GrandietTape collects all the operations that are executed inside it.
    # Then this operations are used to compute the gradients of the loss function with 
    # respect to the trainable variables. Remember that 'persistent=True' is needed
    # iff you want to compute the gradients of the operations inside the tape more than once
    # (for example wrt different losses)

    # Critic training
    # The critic is trained 20 times for each batch
    for i in range(20):
        # Initialize random noise
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape(persistent=True) as tape:
            # Ensure the tape is watching the trainable variables of conditioner
            tape.watch(condition_model.trainable_variables)
            # Step 1: Use condition_model to preprocess conditions and get K values
            # conditions = np.expand_dims(conditions, axis=-1)
            k_values = condition_model(conditions, training=True)
            # Step 2: Generate fake samples using generator
            generated_samples = generator_model([k_values, noise], training=True)
            # Step 3: critic distinguishes real and fake samples
            real_output = critic_model([k_values, real_samples], training=True)
            fake_output = critic_model([k_values, generated_samples], training=True)
            # Step 4: Compute the losses
            critic_loss = wasserstein_loss([real_output, fake_output])
            # Step 5: Compute the gradient penalty
            gp = gradient_penalty(critic_model, real_samples.shape[0], real_samples, generated_samples, k_values)
            # Step 6: Compute the total loss
            critic_loss = critic_loss + 10*gp

        gradients_of_critic = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients_of_critic, critic_model.trainable_variables))

        # Theoretically, weight clipping is a compromise. It's an attempt to enforce a complex mathematical
        # property (Lipschitz continuity, necessary for the Kantorovich-Rubinstein duality to hold) through a 
        # simple, computationally efficient operation (clipping). However, it's worth noting that more sophisticated 
        # methods like gradient penalty and spectral normalization have been proposed in research papers
        # to enforce the Lipschitz condition more effectively.
        # for w in critic_model.trainable_weights:
        #     w.assign(tf.clip_by_value(w, -0.01, 0.01))

    # Delete the tape to free resources
    del tape

    # Generator training
    noise = tf.random.normal([batch_size, latent_dim])
    with tf.GradientTape(persistent=True) as tape:
        # Ensure the tape is watching the trainable variables of conditioner
        tape.watch(condition_model.trainable_variables)
        # Step 1: Use condition_model to preprocess conditions and get K values
        k_values = condition_model(conditions, training=True)
        # Step 2: Generate fake samples using generator
        generated_samples = generator_model([k_values, noise], training=True)
        # Step 3: critic distinguishes real and fake samples
        fake_output = critic_model([k_values, generated_samples], training=True)
        # Compute the losses
        gen_loss = wasserstein_loss(fake_output)

    # Calculate gradients
    gradients_of_generator = tape.gradient(gen_loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    # Apply gradients to update weights
    gradients_of_conditioner = tape.gradient([gen_loss, critic_loss], condition_model.trainable_variables)
    conditioner_optimizer.apply_gradients(zip(gradients_of_conditioner, condition_model.trainable_variables))

    # Delete the tape to free resources
    del tape

    if j % 100 == 0:
        logging.info(f'Epoch: {epoch} | Batch: {j}/{num_batches} | Disc loss: {critic_loss:.5f} | Gen loss: {gen_loss:.5f} | <Score_r>: {real_output.numpy()[1,:].mean():.5f} | <Score_f>: {fake_output.numpy()[1,:].mean():.5f}')

    if j % 500 == 0:
        summarize_performance(real_output, fake_output, critic_loss, gen_loss, generated_samples, real_samples, metrics, j, epoch)
        condition_model.save(f'models/{job_id}/condition_model.h5')
        generator_model.save(f'models/{job_id}/generator_model.h5')
        critic_model.save(f'models/{job_id}/critic_model.h5')
    return condition_model, generator_model, critic_model

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

def summarize_performance(real_output, fake_output, critic_loss, gen_loss, generated_samples, real_samples, metrics, i, epoch):
    '''Summarize the performance of the GAN.
    
    Parameters
    ----------
    real_output : numpy.ndarray
        The output of the critic for the real samples.
    fake_output : numpy.ndarray
        The output of the critic for the fake samples.
    critic_loss : float
        The critic loss.
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
    features = ['Time', 'Event type', 'Size', 'Price', 'Direction']

    # add the metrics to the dictionary
    metrics['critic_loss'].append(critic_loss.numpy())
    metrics['gen_loss'].append(gen_loss.numpy())
    for score in real_output[1,:]:
        metrics['real_score'].append(score)
    for score in fake_output[1,:]:
        metrics['fake_score'].append(score)

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['critic_loss'], label='Critic loss')
    plt.plot(metrics['gen_loss'], label='Generator loss')
    plt.xlabel('Batch x 10')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{job_id}/losses.png')

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_score'], label='Score real')
    plt.plot(metrics['fake_score'], label='Score fake')
    plt.xlabel('Batch x 10')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'plots/{job_id}/score.png')

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

    # Read the features dataframes
    features_paths = [path for path in dataframes_paths if 'features' in path]
    features_paths.sort()
    features_dfs = [pd.read_parquet(f'../data/{stock}_{date}/{path}').iloc[index-1000] for path, index in zip(features_paths, indexes)][:N]

    for i in range(N):
        assert message_dfs[i].shape[0] == features_dfs[i].shape[0], f"The first shapes of message_dfs[{i}] and features_dfs[{i}] are not equal"

    window_size = 1000
    condition_length = int(window_size*0.75)
    input_length = window_size - condition_length
    n_features_input = message_dfs[0].shape[1]
    n_features_condition = features_dfs[0].shape[1]
    num_pieces = 20

    for day in range(N):
        logging.info(f'######################### START DAY {day+1}/{N} #########################')

        # CONDITION DATA
        data_condition = features_dfs[day].values
        # Divide condition data into overlapping pieces
        sub_data = divide_into_overlapping_pieces(data_condition, window_size, num_pieces)
        if not os.path.exists(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy'):
            logging.info('\n[Condition] ---------- PREPROCESSING ----------')
            # The purpose of this preprocessing step is to transform the condition data to have zero mean and unit variance.
            # When you use partial_fit, you don't have access to the entire dataset all at once, but you can still 
            # calculate running estimates for mean and variance based on the data chunks you've seen so far. 
            # This is often done using Welford's algorithm for numerical stability, or a similar online algorithm. 
            # The running estimates are updated as each new chunk of data becomes available.

            # Create a scaler object to scale the condition data
            scaler = StandardScaler()
            num_windows = 0
            logging.info(f'Dividing the condition data into windows and memorize "sub" estimates of mean and variance...')
            for piece_idx, data in enumerate(sub_data):

                if sub_data[-1].shape[0] < window_size:
                     raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # Each piece is divided into windows
                windows = np.array(divide_into_windows(data, window_size))
                # For each window, I have to consider the first condition_length events as condition.
                # The remaining events are the input but are taken from the message file.
                windows = windows[:, :condition_length, :]
                logging.info(f'\twindows shape: {windows.shape}')  # Expected (num_windows, condition_length, n_features_condition)
                num_windows += windows.shape[0]
                # The scaler is updated with the current piece
                scaler.partial_fit(windows.reshape(-1, windows.shape[-1]))
            logging.info('Done.')

            logging.info(f'Total number of windows (condition part): {num_windows}')

            # Create a memmap to store the scaled data. The first shape is sum_i num_windows_i, where num_windows_i is the number of windows
            # for the i-th piece.
            final_shape_condition = (num_windows, condition_length, n_features_condition)
            fp_condition = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape_condition)

            start_idx = 0
            logging.info(f'\nStart scaling on the condition data...')
            for piece_idx, data in enumerate(sub_data):
                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # Here the scaling is performed and the resulting scaled data is assign to the vector fp_condition
                # Each piece is divided into windows and those windows are scaled
                windows = np.array(divide_into_windows(data, window_size))
                windows = windows[:, :condition_length, :]
                logging.info(f'\twindows shape: {windows.shape}')
                scaled_windows = scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape)
                end_idx = start_idx + scaled_windows.shape[0]
                fp_condition[start_idx:end_idx] = scaled_windows
                start_idx = end_idx
                del scaled_windows  # Explicit deletion
            logging.info('Done.')

            logging.info('\nSplit the condition data into train, validation and test sets...')
            condition_train, condition_test = fp_condition[:int(fp_condition.shape[0]*0.75)], fp_condition[int(fp_condition.shape[0]*0.75):]
            condition_train, condition_val = condition_train[:int(condition_train.shape[0]*0.75)], condition_train[int(condition_train.shape[0]*0.75):]
            logging.info('Done.')

            logging.info('\nSave the files...')
            np.save(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', condition_train)
            np.save(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', condition_val)
            np.save(f'../data/condition_test_{stock}_{window_size}_{day+1}.npy', condition_test)
            logging.info('Done.')

            logging.info('\n[Condition] ---------- DONE ----------')
        else:
            logging.info('Loading condition_train, condition_validation and condition_test sets...')
            condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_test = np.load(f'../data/condition_test_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            logging.info('Done.')
            # logging.info('Loading the scaler...')
            # scaler = load(f'tmp/scaler_{stock}_{window_size}_{day+1}_{piece_idx}.joblib')
            # logging.info('Done.')

        # MESSAGE DATA (i.e. the real sample input data)
        data_input = message_dfs[day].values
        # Divide input data into overlapping pieces
        sub_data = divide_into_overlapping_pieces(data_input, window_size, num_pieces)

        if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{day+1}.npy'):
            logging.info('\n[Input] ---------- PREPROCESSING ----------')

            # Create a memmap to store the data. The first shape is the number of windows (samples) for each piece
            # multiplied by the number of pieces.
            final_shape_input = (num_windows, input_length, n_features_input)
            fp_input = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape_input)

            logging.info(f'Dividing the input data into windows')
            start_idx = 0
            for piece_idx, data in enumerate(sub_data):

                if sub_data[-1].shape[0] < window_size:
                    raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

                logging.info(f'\t{piece_idx+1}/{num_pieces}')
                # Each piece is divided into windows
                windows = np.array(divide_into_windows(data, window_size))
                windows = windows[:, condition_length:, :]
                end_idx = start_idx + windows.shape[0]
                fp_input[start_idx:end_idx] = windows
                start_idx = end_idx
                del windows  # Explicit deletion
            logging.info('Done.')

            logging.info('\nSplit the input data into train, validation and test sets...')
            input_train, input_test = fp_input[:int(fp_input.shape[0]*0.75)], fp_input[int(fp_input.shape[0]*0.75):]
            input_train, input_val = input_train[:int(input_train.shape[0]*0.75)], input_train[int(input_train.shape[0]*0.75):]
            logging.info('Done.')

            logging.info('\nSave the files...')
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
            logging.info('Done.')

        # Define the parameters of the GAN.
        # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
        latent_dim = 100
        n_epochs = 100
        T_condition = condition_train.shape[1]
        T_real = input_train.shape[1]
        n_units_generator = 100
        batch_size = 64
        # output condition, hidden units condition, filter dim, kernel size
        gen_units = [5, 64, 5, 25]
        disc_units = [gen_units[0], 64, 5, 25]

        # Use logging.info to print all the hyperparameters
        logging.info(f'\nHYPERPARAMETERS:\n\tlatent_dim: {latent_dim}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_real: {T_real}\n\tFeatures Condition: {features_dfs[day].columns}\n\tFeatures Input: {message_dfs[day].columns}\n\tbatch_size: {batch_size}\n\tnum_batches: {input_train.shape[0]//batch_size}')

        conditioner_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        optimizer = [conditioner_optimizer, generator_optimizer, critic_optimizer]

        generator_model, condition_model = build_generator(T_condition, latent_dim, gen_units, T_real, n_features_condition, n_features_input)
        critic_model = build_critic(T_real, T_condition, n_features_input, disc_units)

        logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')

        condition_model.summary(print_fn=logging.info)
        logging.info('\n')
        generator_model.summary(print_fn=logging.info)
        logging.info('\n')
        critic_model.summary(print_fn=logging.info)
        logging.info('[Model] ---------- DONE ----------\n')
        # exit()

        # Define a dictionary to store the metrics
        metrics = {'critic_loss': [], 'gen_loss': [], 'real_score': [], 'fake_score': []}

        # Train the GAN. When I load the data, I use the argument mmap_mode='r' to avoid to load the data in memory.
        # This is because the data is too big to fit in memory. This means that the data is loaded in memory only when
        # when it is needed.
        num_subvectors = 5
        slice = int(input_train.shape[0] / num_subvectors)
        for i in range(num_subvectors):
            logging.info(f'\n---------- Training on piece {i} ----------')
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
                condition_model = tf.keras.models.load_model(f'models/{job_id}/condition_model.h5')
                generator_model = tf.keras.models.load_model(f'models/{job_id}/generator_model.h5')
                critic_model = tf.keras.models.load_model(f'models/{job_id}/critic_model.h5')
            for epoch in range(n_epochs):
                j = 0
                for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
                    j += 1
                    batch_size = batch_real_samples.shape[0]
                    condition_model, generator_model, critic_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, batch_size, T_condition, latent_dim, j, epoch, metrics)
            # save the models
            condition_model.save(f'models/{job_id}/condition_model.h5')
            generator_model.save(f'models/{job_id}/generator_model.h5')
            critic_model.save(f'models/{job_id}/critic_model.h5')

        # Remeber to handle the last piece
        logging.info(f'---------- Training on the LAST piece ----------')
        input_train = input_train[(i+1)*slice:]
        condition_train = condition_train[(i+1)*slice:]
        dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
        condition_model = tf.keras.models.load_model(f'models/{job_id}/condition_model.h5')
        generator_model = tf.keras.models.load_model(f'models/{job_id}/generator_model.h5')
        critic_model = tf.keras.models.load_model(f'models/{job_id}/critic_model.h5')
        for epoch in range(n_epochs):
            i = 0
            for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
                i += 1
                condition_model, generator_model, critic_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, T_condition, latent_dim, i, epoch, metrics)
        # save the models
        condition_model.save(f'models/{job_id}/condition_model.h5')
        generator_model.save(f'models/{job_id}/generator_model.h5')
        critic_model.save(f'models/{job_id}/critic_model.h5')

        logging.info(f'##################### END DAY {day+1}/{N} #####################\n')