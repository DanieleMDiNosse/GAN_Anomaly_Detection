'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from data_utils import *
# import psutil
# import time
# from metrics import return_distribution, volatility, bid_ask_spread
import argparse
import logging
# from tensorflow.keras.utils import plot_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tqdm import tqdm
from statsmodels.graphics.tsaplots import plot_acf
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
    # condition_model.compile(optimizer='adam', loss='mean_squared_error')
    # plot_model(condition_model, to_file=f'plots/{os.getpid()}/condition_model_plot.png', show_shapes=True, show_layer_names=True)

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

    # generator_model.compile(optimizer='adam', loss='mean_squared_error')
    # plot_model(generator_model, to_file=f'plots/{os.getpid()}/generator_model_plot.png', show_shapes=True, show_layer_names=True)
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
    # plot_model(critic_model, to_file=f'plots/{os.getpid()}/critic_model_plot.png', show_shapes=True, show_layer_names=True)

    return critic_model

def train_step(real_samples, conditions, condition_model, generator_model, critic_model, optimizer, T_cond, latent_dim, i, epoch, metrics):
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

    if i % 200 == 0:
        summarize_performance(real_output, fake_output, critic_loss, gen_loss, generated_samples, metrics)
        condition_model.save(f'models/{os.getpid()}/condition_model.h5')
        generator_model.save(f'models/{os.getpid()}/generator_model.h5')
        critic_model.save(f'models/{os.getpid()}/critic_model.h5')
        # Plot the generated samples
        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.plot(generated_samples[0, :, 0], label='Ask price')
        plt.plot(generated_samples[0, :, 1], label='Ask volume')
        plt.xlabel('Time (Events)')
        plt.title('Generated sample')
        plt.legend()
        plt.savefig(f'plots/{os.getpid()}/generated_samples_{epoch}_{i}.png')
        plt.savefig(f'plots/{os.getpid()}/generated_samples_{epoch}_{i}.png')
        plt.close()
    if i % 500 == 0:
        logging.info(f'Saving the generated samples. Update of stat metrics...')
        np.save(f'generated_samples/{os.getpid()}/generated_samples.npy', generated_samples)
        # Load all the generated samples files that are in the generated_samples folder and stack them one after the other
        generated_samples = np.vstack([np.load(f'generated_samples/{os.getpid()}/{file}').reshape(generated_samples.shape[0]*generated_samples.shape[1], generated_samples.shape[2]) \
                                       for file in os.listdir(f'generated_samples/{os.getpid()}/')])
        # print(generated_samples.shape)
        # return_distribution(generated_samples)
        # volatility(generated_samples)
        # bid_ask_spread(generated_samples)
        # plt.close()
        # print(generated_samples.shape)
        # return_distribution(generated_samples)
        # volatility(generated_samples)
        # bid_ask_spread(generated_samples)
        # plt.close()

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
    '''Compute the Wasserstein loss.
    
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
        w_tot = -backend.mean(fake_output*fake_labels)
    return w_tot

def summarize_performance(real_output, fake_output, critic_loss, gen_loss, generated_samples, metrics):
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
    plt.savefig(f'plots/{os.getpid()}/losses.png')

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_score'], label='Score real')
    plt.plot(metrics['fake_score'], label='Score fake')
    plt.xlabel('Batch x 10')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(f'plots/{os.getpid()}/score.png')

    # Plot the fake samples together with the real samples
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(generated_samples[0, :, 0], label='Ask price')
    plt.plot(generated_samples[0, :, 1], label='Ask volume')
    plt.xlabel('Time (Events)')
    plt.title('Generated sample')
    plt.legend()
    plt.savefig(f'plots/{os.getpid()}/generated_samples.png')

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
    logging.basicConfig(filename=f'output_{os.getpid()}', format='%(message)s', level=levels[args.log])

    logger = tf.get_logger()
    logger.setLevel('ERROR')

    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time: {formatted_datetime}")

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
        for device in physical_devices:
            logging.info(f'{device}\n')
    
    # Read the orderbook dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    dataframes_paths = [path for path in dataframes_paths if 'orderbook' in path]
    dataframes_paths.sort()
    dataframes = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in dataframes_paths][:N]

    window_size = 500
    levels = 4
    condition_length = int(window_size*0.8)
    input_length = window_size - condition_length
    for day in range(len(dataframes)):
        logging.info(f'##################### START DAY {day+1}/{len(dataframes)} #####################')
        data = dataframes[day].values[:,:levels]
        # Convert data entries into int32 to avoid memory issues
        data = data.astype(np.int32)

        logging.info('Evaluate the returns, the volatility, the imbalance and the bid-ask spread...')
        p_a, p_b = data[:,0], data[:,2]
        returns = np.diff(p_a)
        returns = np.insert(returns, 0, 0)
        volatilty = np.std(p_a)
        imbalance = evaluate_imbalance(data, levels)
        spread = evaluate_spread(p_b, p_a)
        logging.info('Done.')

        # Create a new array to store the features
        data = np.empty(shape=(data.shape[0], 4))
        data[:,0] = returns
        data[:,1] = volatilty
        data[:,2] = imbalance
        data[:,3] = spread

        # Compute the autocorreltions
        logging.info('Compute the lag in which the autocorrelations are negligible...')
        autocorrelations = []
        for i in range(data.shape[1]):
            autocorrelations.append(fast_autocorrelation(data[:,i])[1])
            # plot_acf(data[:,i], lags=data[:,i].shape[0])
            # plt.savefig(f'plots/{os.getpid()}/autocorrelation_{i}.png')
        logging.info(f'Autocorrelations: {autocorrelations}')
        logging.info('Done.')
        exit()
        
        if not os.path.exists(f'../data/input_train_{stock}_{window_size}_{levels}_{day+1}.npy'):
            logging.info('\n---------- PREPROCESSING ----------')
            scaler = StandardScaler()
            try:
                # Create a memmap to store the data. The first shape is the number of sample for each piece
                # multiplied by the number of pieces.
                final_shape = (2008536, window_size, levels)
                fp = np.memmap("final_data.dat", dtype='int32', mode='w+', shape=final_shape)
                # Divide data in 10 pieces
                sub_data = np.array_split(data, 10)
                start_idx = 0
                for i, v in enumerate(sub_data):
                    logging.info(f'Dividing the data into windows and memorize "sub" quantites - {i+1}/10')
                    # Each piece is divided into windows
                    windows = np.array(divide_into_windows(v, window_size))
                    np.save(f'tmp/windows_{i}.npy', windows)
                    # Using partial_fit, scaler learn incrementally the quantities needed for the scaling process. 
                    # This allows to scale the data without loading it all together in memory.
                    windows_mmap = np.load(f'tmp/windows_{i}.npy', mmap_mode='r')
                    scaler.partial_fit(windows_mmap.reshape(-1, windows_mmap.shape[-1]))
                    logging.info('Done.')

                # Here the scaling is performed and the resulting scaled data is assign to the vector fp
                logging.info('\nStart scaling...')
                for i in range(10):
                    windows_mmap = np.load(f'tmp/windows_{i}.npy', mmap_mode='r')
                    scaled_windows = scaler.transform(windows_mmap.reshape(-1, windows_mmap.shape[-1])).reshape(windows_mmap.shape)
                    end_idx = start_idx + scaled_windows.shape[0]
                    fp[start_idx:end_idx] = scaled_windows
                    start_idx = end_idx
                logging.info('Done.')
                
                logging.info('\nSplit the data into train, validation and test sets...')
                train, test = fp[:int(fp.shape[0]*0.75)], fp[int(fp.shape[0]*0.75):]
                train, val = train[:int(train.shape[0]*0.75)], train[int(train.shape[0]*0.75):]
                np.save(f'../data/train_{stock}_{window_size}_{levels}_{day+1}.npy', train)
                np.save(f'../data/val_{stock}_{window_size}_{levels}_{day+1}.npy', val)
                np.save(f'../data/test_{stock}_{window_size}_{levels}_{day+1}.npy', test)
                logging.info('Done.')

                logging.info('\nDivide the data into conditions and input...')
                condition_train, input_train = divide_data_condition_input(train, condition_length)
                condition_val, input_val = divide_data_condition_input(val, condition_length)
                condition_test, input_test = divide_data_condition_input(test, condition_length)
                logging.info('Done.')

                logging.info('\nSave all the preprocessed data...')
                np.save(f'../data/condition_train_{stock}_{window_size}_{levels}_{day+1}.npy', condition_train)
                np.save(f'../data/condition_val_{stock}_{window_size}_{levels}_{day+1}.npy', condition_val)
                np.save(f'../data/condition_test_{stock}_{window_size}_{levels}_{day+1}.npy', condition_test)
                np.save(f'../data/input_train_{stock}_{window_size}_{levels}_{day+1}.npy', input_train)
                np.save(f'../data/input_val_{stock}_{window_size}_{levels}_{day+1}.npy', input_val)
                np.save(f'../data/input_test_{stock}_{window_size}_{levels}_{day+1}.npy', input_test)
                logging.info('Done.')
                logging.info('\n---------- DONE ----------')
            except Exception as e:
                logging.error(f'Error: {e}')
                exit()
        else:
            logging.info('Loading train, validation and test sets...')
            input_train = np.load(f'../data/input_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            input_val = np.load(f'../data/input_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            input_test = np.load(f'../data/input_test_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            condition_test = np.load(f'../data/condition_test_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
            logging.info('Done.')

        logging.info(f'{type(input_train)}')
        # Folders creation
        os.mkdir(f'plots/{os.getpid()}') # Model architecture plots, metrics plots
        os.mkdir(f'generated_samples/{os.getpid()}') # Generated samples
        os.mkdir(f'models/{os.getpid()}') # Models

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
        logging.info(f'HYPERPARAMETERS:\n\tlatent_dim: {latent_dim}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_real: {T_real}\n\tLevels: {levels}\n\tbatch_size: {batch_size}')

        # Create a TensorFlow Dataset object
        #dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)

        conditioner_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
        optimizer = [conditioner_optimizer, generator_optimizer, critic_optimizer]

        generator_model, condition_model = build_generator(T_condition, latent_dim, gen_units, T_real, levels)
        critic_model = build_critic(T_real, T_condition, levels, disc_units)

        # Define a dictionary to store the metrics
        metrics = {'critic_loss': [], 'gen_loss': [], 'real_score': [], 'fake_score': []}

        # Train the GAN. When I load the data, I use the argumento mmap_mode='r' to avoid to load the data in memory.
        # This is because the data is too big to fit in memory. This means that the data is loaded in memory only when
        # when it is needed.
        num_subvectors = 5
        slice = int(input_train.shape[0] / num_subvectors)
        for i in range(num_subvectors):
            logging.info(f'---------- Training on piece {i} ----------')
            input_train = input_train[i*slice: (i+1)*slice]
            condition_train = condition_train[i*slice: (i+1)*slice]
            dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
            if i > 0:
                # Load the models at the end of the previous training
                condition_model = tf.keras.models.load_model(f'models/{os.getpid()}/condition_model.h5')
                generator_model = tf.keras.models.load_model(f'models/{os.getpid()}/generator_model.h5')
                critic_model = tf.keras.models.load_model(f'models/{os.getpid()}/critic_model.h5')
            for epoch in range(n_epochs):
                i = 0
                for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
                    i += 1
                    condition_model, generator_model, critic_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, T_condition, latent_dim, i, epoch, metrics)
            # save the models
            condition_model.save(f'models/{os.getpid()}/condition_model.h5')
            generator_model.save(f'models/{os.getpid()}/generator_model.h5')
            critic_model.save(f'models/{os.getpid()}/critic_model.h5')

        # Remeber to handle the last piece
        logging.info(f'---------- Training on the LAST piece ----------')
        input_train = input_train[(i+1)*slice:]
        condition_train = condition_train[(i+1)*slice:]
        dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
        condition_model = tf.keras.models.load_model(f'models/{os.getpid()}/condition_model.h5')
        generator_model = tf.keras.models.load_model(f'models/{os.getpid()}/generator_model.h5')
        critic_model = tf.keras.models.load_model(f'models/{os.getpid()}/critic_model.h5')
        for epoch in range(n_epochs):
            i = 0
            for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
                i += 1
                condition_model, generator_model, critic_model = train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, T_condition, latent_dim, i, epoch, metrics)
        # save the models
        condition_model.save(f'models/{os.getpid()}/condition_model.h5')
        generator_model.save(f'models/{os.getpid()}/generator_model.h5')
        critic_model.save(f'models/{os.getpid()}/critic_model.h5')
        logging.info(f'##################### END DAY {day+1}/{len(dataframes)} #####################')