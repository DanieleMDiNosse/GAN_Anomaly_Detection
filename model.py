'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import *
from joblib import dump, load
import argparse
import logging
from tensorflow.keras.utils import plot_model
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, TimeDistributed, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend


def build_generator(T_cond, latent_dim, gen_units, T_real, num_features):
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
    condition_input = Input(shape=(T_cond, num_features))
    lstm = LSTM(hidden_units0, return_sequences=True)(condition_input)
    dense_layer = Dense(K, activation='relu')
    output = TimeDistributed(dense_layer)(lstm)
    output = Dropout(0.2)(output)
    condition_model = Model(condition_input, output)
    plot_model(condition_model, to_file=f'plots/{job_id}/condition_model_plot.png', show_shapes=True, show_layer_names=True)

    # ----------------- GENERATOR -----------------
    hidden_units1 = gen_units[2]
    condition_input = Input(shape=(T_cond, K,))
    noise_input = Input(shape=(T_cond, latent_dim,))
    input = Concatenate()([condition_input, noise_input])
    lstm = LSTM(hidden_units1, return_sequences=True)(input)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(hidden_units1)(lstm)
    dense = Dense(num_features*T_real, activation='linear')(lstm)
    dropout = Dropout(0.2)(dense)
    reshape = Reshape((T_real, num_features))(dropout)
    generator_model = Model([condition_input, noise_input], reshape)
    plot_model(generator_model, to_file=f'plots/{job_id}/generator_model_plot.png', show_shapes=True, show_layer_names=True)
    return generator_model, condition_model

def build_critic(T_real, T_cond, num_features, disc_units):
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
    condition_input = Input(shape=(T_cond, K,))
    lstm = LSTM(hidden_units0)(condition_input)
    dense_layer = Dense(T_real*num_features, activation='leaky_relu')(lstm)
    reshape = Reshape((T_real, num_features))(dense_layer)

    # Input for the real samples
    hidden_units1 = disc_units[1]
    input = Input(shape=(T_real, num_features))
    concat = Concatenate()([reshape, input])

    lstm = LSTM(hidden_units1)(concat)
    lstm = Dropout(0.2)(lstm)
    output = Dense(1, activation='linear')(lstm)

    critic_model = Model([condition_input, input], output)
    plot_model(critic_model, to_file=f'plots/{job_id}/critic_model_plot.png', show_shapes=True, show_layer_names=True)

    return critic_model

# @tf.function
def train_step(real_samples, conditions, condition_model, generator_model, critic_model, optimizer, T_cond, latent_dim, i, epoch, metrics, scaler):
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

    # Initialize random noise
    noise = tf.random.normal([batch_conditions.shape[0], T_cond, latent_dim])

    # Create a GradientTape for the conditioner, generator, and critic.
    # GrandietTape collects all the operations that are executed inside it.
    # Then this operations are used to compute the gradeints of the loss function with 
    # respect to the trainable variables.

    # Critic training
    # The critic is trained 5 times for each batch
    for _ in range(5):
        with tf.GradientTape(persistent=True) as tape:
            # Ensure the tape is watching the trainable variables of conditioner
            tape.watch(condition_model.trainable_variables)
            # Step 1: Use condition_model to preprocess conditions and get K values
            k_values = condition_model(conditions, training=True)
            # Step 2: Generate fake samples using generator
            generated_samples = generator_model([k_values, noise], training=True)
            # Step 3: critic distinguishes real and fake samples
            real_output = critic_model([k_values, real_samples], training=True)
            fake_output = critic_model([k_values, generated_samples], training=True)
            # Compute the losses
            critic_loss = wasserstein_loss([real_output, fake_output])

        gradients_of_critic = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(zip(gradients_of_critic, critic_model.trainable_variables))

        # Theoretically, weight clipping is a compromise. It's an attempt to enforce a complex mathematical
        # property (Lipschitz continuity, necessary for the Kantorovich-Rubinstein duality to hold) through a 
        # simple, computationally efficient operation (clipping). However, it's worth noting that more sophisticated 
        # methods like gradient penalty and spectral normalization have been proposed in subsequent research 
        # to enforce the Lipschitz condition more effectively.
        for w in critic_model.trainable_weights:
            w.assign(tf.clip_by_value(w, -0.01, 0.01))

    # Delete the tape to free resources
    del tape

    # Generator training
    noise = tf.random.normal([batch_real_samples.shape[0], T_cond, latent_dim])
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

    if i % 100 == 0:
        summarize_performance(real_output, fake_output, critic_loss, gen_loss, generated_samples, real_samples, metrics, i, epoch)
        condition_model.save(f'models/{job_id}/condition_model.h5')
        generator_model.save(f'models/{job_id}/generator_model.h5')
        critic_model.save(f'models/{job_id}/critic_model.h5')
    return condition_model, generator_model, critic_model

def compute_critic_loss(real_output, fake_output):
    '''Compute the critic loss.
    
    Parameters
    ----------
    real_output : numpy.ndarray
        The output of the critic for the real samples.
    fake_output : numpy.ndarray
        The output of the critic for the fake samples.
    
    Returns
    -------
    total_critic_loss : float
        The critic loss.'''

    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_critic_loss = real_loss + fake_loss
    return tf.reduce_mean(total_critic_loss)

def compute_generator_loss(fake_output):
    '''Compute the generator loss.
    
    Parameters
    ----------
    fake_output : numpy.ndarray
        The output of the critic for the fake samples.
    
    Returns
    -------
    float
        The generator loss.'''
    return tf.reduce_mean(binary_crossentropy(tf.ones_like(fake_output), fake_output))

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
        true_labels = -tf.ones_like(real_output)
        fake_labels = tf.ones_like(fake_output)
        w_real = backend.mean(real_output*true_labels)
        w_fake = backend.mean(fake_output*fake_labels)
        w_tot = w_real - w_fake
        w_tot = w_real - w_fake
    else:
        fake_output = predictions
        fake_labels = tf.ones_like(fake_output)
        w_tot = -backend.mean(fake_output*fake_labels)
    return w_tot

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
    features = ['Returns Ask', 'Returns Bid', 'Imbalance', 'Spread', 'Volatility Ask', 'Volatility Bid']

    # add the metrics to the dictionary
    metrics['critic_loss'].append(critic_loss.numpy())
    metrics['gen_loss'].append(gen_loss.numpy())
    for score in real_output[1,:]:
        metrics['real_score'].append(score)
    for score in fake_output[1,:]:
        metrics['fake_score'].append(score)

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['critic_loss'], label='critic loss')
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
        axes[j].plot(scaler.inverse_transform(generated_samples.reshape(-1,generated_samples.shape[-1]).reshape(generated_samples.shape))[:, j], label=f'Generated {feature}')
        axes[j].set_title(f'Generated {feature}')
    axes[j].set_xlabel('Time (Events)')
    plt.savefig(f'plots/{job_id}/generated_samples_{epoch}_{i}.png')

    # Plot a chosen real sample
    fig, axes = plt.subplots(real_samples.shape[1], 1, figsize=(10, 10), tight_layout=True)
    for j, feature in zip(range(real_samples.shape[1]), features):
        axes[j].plot(scaler.inverse_transform(real_samples.reshape(-1,real_samples.shape[-1]).reshape(real_samples.shape))[:, j])
        axes[j].set_title(f'Real {feature}')
    axes[j].set_xlabel('Time (Events)')
    plt.savefig(f'plots/{job_id}/real_samples.png')

    plt.close('all')

    logging.info(f'Epoch: {epoch} | Batch: {i} | Disc loss: {critic_loss:.5f} | Gen loss: {gen_loss:.5f} | <Score_r>: {real_output.numpy()[1,:].mean():.5f} | <Score_f>: {fake_output.numpy()[1,:].mean():.5f}\n')
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
    
    # Read the features dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    dataframes_paths = [path for path in dataframes_paths if 'features' in path]
    dataframes_paths.sort()
    dataframes = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in dataframes_paths][:N]

    window_size = 1500
    condition_length = int(window_size*0.75)
    input_length = window_size - condition_length
    n_features = dataframes[0].shape[1]
    num_pieces = 5

    for day in range(len(dataframes)):
        logging.info(f'######################### START DAY {day+1}/{len(dataframes)} #########################')

        data = dataframes[day].values
        # Divide data in pieces
        print(data.shape)
        sub_data = np.array_split(data, num_pieces)
        for piece_idx, data in enumerate(sub_data):
            logging.info(f'==================== START PIECE {piece_idx+1}/{num_pieces} ====================')
            if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{day+1}_{piece_idx}.npy'):
                logging.info('\n---------- PREPROCESSING ----------')
                # The purpose of this preprocessing step is to transform the data to have zero mean and unit variance.
                # When you use partial_fit, you don't have access to the entire dataset all at once, but you can still 
                # calculate running estimates for mean and variance based on the data chunks you've seen so far. 
                # This is often done using Welford's algorithm for numerical stability, or a similar online algorithm. 
                # The running estimates are updated as each new chunk of data becomes available

                scaler = StandardScaler()
                # Divide data in pieces again
                sub_data = np.array_split(data, num_pieces)

                if sub_data[-1].shape[0] < window_size:
                     raise ValueError(f'The last piece has shape {sub_data[-1].shape} and it is smaller than the window size {window_size}.')

                num_windows = 0
                for i, v in enumerate(sub_data):
                    logging.info(f'Dividing the data into windows and memorize "sub" quantites - {i+1}/{num_pieces}')
                    # Each piece is divided into windows
                    windows = np.array(divide_into_windows(v, window_size))
                    num_windows += windows.shape[0]
                    # The scaler is updated with the current piece
                    scaler.partial_fit(windows.reshape(-1, windows.shape[-1]))
                    logging.info('Done.')
                print(f'Total number of windows: {num_windows}')
                logging.info(f'Total number of windows: {num_windows}')

                # Create a memmap to store the data. The first shape is the number of windows (samples) for each piece
                # multiplied by the number of pieces.
                final_shape = (num_windows, window_size, n_features)
                fp = np.memmap("final_data.dat", dtype='float32', mode='w+', shape=final_shape)

                # Here the scaling is performed and the resulting scaled data is assign to the vector fp
                logging.info('\nStart scaling...')
                start_idx = 0
                for i in range(num_pieces):
                    scaled_windows = scaler.transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape)
                    end_idx = start_idx + scaled_windows.shape[0]
                    fp[start_idx:end_idx] = scaled_windows
                    start_idx = end_idx
                    del scaled_windows  # Explicit deletion
                logging.info('Done.')
                
                logging.info('\nDump the scaler...')
                dump(scaler, f'tmp/scaler_{stock}_{window_size}_{day+1}.joblib')
                logging.info('Done.')

                logging.info('\nSplit the data into train, validation and test sets...')
                train, test = fp[:int(fp.shape[0]*0.75)], fp[int(fp.shape[0]*0.75):]
                train, val = train[:int(train.shape[0]*0.75)], train[int(train.shape[0]*0.75):]
                np.save(f'../data/train_{stock}_{window_size}_{day+1}.npy', train)
                np.save(f'../data/val_{stock}_{window_size}_{day+1}.npy', val)
                np.save(f'../data/test_{stock}_{window_size}_{day+1}.npy', test)
                logging.info('Done.')

                logging.info('\nDivide the data into conditions and input...')
                condition_train = np.memmap('condition_train.dat', dtype='float32', mode='w+', shape=(train.shape[0], condition_length, n_features))
                input_train = np.memmap('input_train.dat', dtype='float32', mode='w+', shape=(train.shape[0], input_length, n_features))
                sub_train = np.array_split(train, num_pieces)
                start_idx = 0
                for i, v in enumerate(sub_train):
                    logging.info(f'Dividing the train data into conditions and input - {i+1}/{num_pieces}')
                    condition, input = parallel_divide_data(v, condition_length)
                    end_idx = start_idx + condition.shape[0]
                    condition_train[start_idx:end_idx] = condition
                    input_train[start_idx:end_idx] = input
                    start_idx = end_idx
                    del condition, input
                gc.collect()
                
                condition_val = np.memmap('condition_val.dat', dtype='float32', mode='w+', shape=(val.shape[0], condition_length, n_features))
                input_val = np.memmap('input_val.dat', dtype='float32', mode='w+', shape=(val.shape[0], input_length, n_features))
                sub_val = np.array_split(val, num_pieces)
                start_idx = 0
                for i, v in enumerate(sub_val):
                    logging.info(f'Dividing the validation data into conditions and input - {i+1}/{num_pieces}')
                    condition, input = parallel_divide_data(v, condition_length)
                    end_idx = start_idx + condition.shape[0]
                    condition_val[start_idx:end_idx] = condition
                    input_val[start_idx:end_idx] = input
                    start_idx = end_idx
                    del condition, input
                gc.collect()

                condition_test = np.memmap('condition_test.dat', dtype='float32', mode='w+', shape=(test.shape[0], condition_length, n_features))
                input_test = np.memmap('input_test.dat', dtype='float32', mode='w+', shape=(test.shape[0], input_length, n_features))
                sub_test = np.array_split(test, num_pieces)
                start_idx = 0
                for i, v in enumerate(sub_test):
                    logging.info(f'Dividing the test data into conditions and input - {i+1}/{num_pieces}')
                    condition, input = parallel_divide_data(v, condition_length)
                    end_idx = start_idx + condition.shape[0]
                    condition_test[start_idx:end_idx] = condition
                    input_test[start_idx:end_idx] = input
                    start_idx = end_idx
                    del condition, input
                logging.info('Done.')
                gc.collect()

                logging.info('\nSave all the preprocessed data...')
                np.save(f'../data/condition_train_{stock}_{window_size}_{day+1}_{piece_idx}.npy', condition_train)
                np.save(f'../data/condition_val_{stock}_{window_size}_{day+1}_{piece_idx}.npy', condition_val)
                np.save(f'../data/condition_test_{stock}_{window_size}_{day+1}_{piece_idx}.npy', condition_test)
                np.save(f'../data/input_train_{stock}_{window_size}_{day+1}_{piece_idx}.npy', input_train)
                np.save(f'../data/input_val_{stock}_{window_size}_{day+1}_{piece_idx}.npy', input_val)
                np.save(f'../data/input_test_{stock}_{window_size}_{day+1}_{piece_idx}.npy', input_test)
                logging.info('Done.')
                logging.info('\n---------- DONE ----------')
            else:
                logging.info('Loading train, validation and test sets...')
                input_train = np.load(f'../data/input_train_{stock}_{window_size}_{day+1}_{piece_idx}.npy', mmap_mode='r')
                input_val = np.load(f'../data/input_val_{stock}_{window_size}_{day+1}_{piece_idx}.npy', mmap_mode='r')
                input_test = np.load(f'../data/input_test_{stock}_{window_size}_{day+1}_{piece_idx}.npy', mmap_mode='r')
                condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{day+1}_{piece_idx}.npy', mmap_mode='r')
                condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{day+1}_{piece_idx}.npy', mmap_mode='r')
                condition_test = np.load(f'../data/condition_test_{stock}_{window_size}_{day+1}_{piece_idx}.npy', mmap_mode='r')
                logging.info('Done.')
                logging.info('Loading the scaler...')
                scaler = load(f'tmp/scaler_{stock}_{window_size}_{day+1}_{piece_idx}.joblib')
                logging.info('Done.')

            # Define the parameters of the GAN.
            # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
            latent_dim = 100
            n_epochs = 100
            T_condition = condition_train.shape[1]
            T_real = input_train.shape[1]
            n_units_generator = 100
            batch_size = 64
            gen_units = [5, 64, 64]
            disc_units = [gen_units[0], 64, 64]

            # Use logging.info to print all the hyperparameters
            logging.info(f'HYPERPARAMETERS:\n\tlatent_dim: {latent_dim}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_real: {T_real}\n\tFeatures: {dataframes[day].columns}\n\tbatch_size: {batch_size}')

            conditioner_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
            optimizer = [conditioner_optimizer, generator_optimizer, critic_optimizer]

            generator_model, condition_model = build_generator(T_condition, latent_dim, gen_units, T_real, n_features)
            critic_model = build_critic(T_real, T_condition, n_features, disc_units)
            exit()

            # Define a dictionary to store the metrics
            metrics = {'critic_loss': [], 'gen_loss': [], 'real_score': [], 'fake_score': []}

            # Train the GAN. When I load the data, I use the argument mmap_mode='r' to avoid to load the data in memory.
            # This is because the data is too big to fit in memory. This means that the data is loaded in memory only when
            # when it is needed.
            num_subvectors = 5
            slice = int(input_train.shape[0] / num_subvectors)
            for i in range(num_subvectors):
                logging.info(f'---------- Training on piece {i} ----------')
                input_train = input_train[i*slice: (i+1)*slice]
                condition_train = condition_train[i*slice: (i+1)*slice]
                dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
                if i > 0 or piece_idx > 0:
                    # Load the models of the previous training (previous piece)
                    condition_model = tf.keras.models.load_model(f'models/{job_id}/condition_model.h5')
                    generator_model = tf.keras.models.load_model(f'models/{job_id}/generator_model.h5')
                    critic_model = tf.keras.models.load_model(f'models/{job_id}/critic_model.h5')
                for epoch in range(n_epochs):
                    j = 0
                    for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
                        j += 1
                        condition_model, generator_model, critic_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, T_condition, latent_dim, j, epoch, metrics, scaler)
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
                    condition_model, generator_model, critic_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, T_condition, latent_dim, i, epoch, metrics, scaler)
            # save the models
            condition_model.save(f'models/{job_id}/condition_model.h5')
            generator_model.save(f'models/{job_id}/generator_model.h5')
            critic_model.save(f'models/{job_id}/critic_model.h5')
            logging.info(f'==================== END PIECE {piece_idx+1}/{num_pieces} ====================')
        logging.info(f'##################### END DAY {day+1}/{len(dataframes)} #####################')