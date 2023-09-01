'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import divide_into_windows, divide_data_condition_input
import argparse
import logging
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
    condition_model = Model(condition_input, output)
    condition_model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(condition_model, to_file=f'plots/condition_model_plot_{os.getpid()}.png', show_shapes=True, show_layer_names=True)

    # ----------------- GENERATOR -----------------
    hidden_units1 = gen_units[2]
    condition_input = Input(shape=(T_cond, K,))
    noise_input = Input(shape=(T_cond, latent_dim,))
    input = Concatenate()([condition_input, noise_input])
    lstm = LSTM(hidden_units1, return_sequences=True)(input)
    lstm = LSTM(hidden_units1)(lstm)
    dense = Dense(num_features*T_real, activation='linear')(lstm)
    dropout = Dropout(0.2)(dense)
    reshape = Reshape((T_real, num_features))(dropout)
    generator_model = Model([condition_input, noise_input], reshape)

    generator_model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(generator_model, to_file=f'plots/generator_model_plot_{os.getpid()}.png', show_shapes=True, show_layer_names=True)
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
    output = Dense(1, activation='linear')(lstm)

    critic_model = Model([condition_input, input], output)
    critic_model.compile(loss='binary_crossentropy', optimizer='adam')
    # plot the models
    plot_model(critic_model, to_file=f'plots/critic_model_plot_{os.getpid()}.png', show_shapes=True, show_layer_names=True)

    return critic_model

def train_step(real_samples, conditions, condition_model, generator_model, critic_model, optimizer, T_cond, latent_dim, i, metrics):
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
    None'''
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

    if i % 3 == 0:
        summarize_performance(real_output, fake_output, critic_loss, gen_loss, generated_samples, metrics)

    return None

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
        w_tot = w_real + w_fake
    else:
        fake_output = predictions
        fake_labels = tf.ones_like(fake_output)
        w_tot = backend.mean(fake_output*fake_labels)
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

    # Calculate critic Accuracy on Real Samples
    real_labels = -np.ones((real_output.shape[0], 1))  # Real samples have label 1
    real_accuracy = accuracy_score(real_labels, np.round(real_output.numpy()))
    
    # Calculate critic Accuracy on Fake Samples
    fake_labels = np.ones((fake_output.shape[0], 1))  # Fake samples have label 0
    fake_accuracy = accuracy_score(fake_labels, np.round(fake_output.numpy()))

    # add the metrics to the dictionary
    metrics['critic_loss'].append(critic_loss.numpy())
    metrics['gen_loss'].append(gen_loss.numpy())
    metrics['real_acc'].append(real_accuracy)
    metrics['fake_acc'].append(fake_accuracy)

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['critic_loss'], label='critic loss')
    plt.plot(metrics['gen_loss'], label='Generator loss')
    plt.xlabel('Batch x 10')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/losses_{os.getpid()}.png')

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_acc'], label='Accuracy real')
    plt.plot(metrics['fake_acc'], label='Accuracy fake')
    plt.xlabel('Batch x 10')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots/accurac_{os.getpid()}.png')

    # Plot the fake samples together with the real samples
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(generated_samples[0, :, 0], label='Ask price')
    plt.plot(generated_samples[0, :, 1], label='Ask volume')
    plt.xlabel('Time (Events)')
    plt.title('Generated sample')
    plt.legend()
    plt.savefig(f'plots/generated_samples_{os.getpid()}.png')

    print(f'Batch: {i} | Disc loss: {critic_loss:.5f} | Gen loss: {gen_loss:.5f} | Real acc: {real_accuracy:.5f} | Fake acc: {fake_accuracy:.5f}')
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
    logging.basicConfig(format='%(message)s', level=levels[args.log])

    # Load the data
    stock = args.stock
    if stock == 'TSLA':
        date = '2015-01-01_2015-01-31_10'
    elif stock == 'MSFT':
        date = '2018-04-01_2018-04-30_5'

    N = args.N_days
    
    # Read the orderbook dataframes
    dataframes_paths = os.listdir(f'../data/{stock}_{date}/')
    dataframes_paths = [path for path in dataframes_paths if 'orderbook' in path]
    dataframes_paths.sort()
    dataframes = [pd.read_parquet(f'../data/{stock}_{date}/{path}') for path in dataframes_paths][:N]

    # For now I will focus just on the first day as a proof of concept.
    window_size = 500
    condition_length = int(window_size*0.7)
    input_length = window_size - condition_length
    data = dataframes[0].values[:,:5]
    print('Data shape: ', data.shape)
    
    if not os.path.exists(f'../data/input_train_{stock}.npy'):
        logging.info('\n---------- PREPROCESSING ----------')
        logging.info('\nDividing the data into windows...')
        input_data = np.array(divide_into_windows(data, window_size))
        logging.info('Scaling the data...')
        scaler = StandardScaler()
        input_data = scaler.fit_transform(input_data.reshape(-1, input_data.shape[-1])).reshape(input_data.shape)
        logging.info('Split the data into train, validation and test sets...')
        train, test = train_test_split(input_data, test_size=0.2, shuffle=False)
        train, val = train_test_split(train, test_size=0.2, shuffle=False)
        np.save(f'../data/train_{stock}.npy', train)
        np.save(f'../data/val_{stock}.npy', val)
        np.save(f'../data/test_{stock}.npy', test)
        logging.info('Divide the data into conditions and input...')
        condition_train, input_train = divide_data_condition_input(train, condition_length)
        condition_val, input_val = divide_data_condition_input(val, condition_length)
        condition_test, input_test = divide_data_condition_input(test, condition_length)
        logging.info('Save all the preprocessed data...')
        np.save(f'../data/condition_train_{stock}.npy', condition_train)
        np.save(f'../data/condition_val_{stock}.npy', condition_val)
        np.save(f'../data/condition_test_{stock}.npy', condition_test)
        np.save(f'../data/input_train_{stock}.npy', input_train)
        np.save(f'../data/input_val_{stock}.npy', input_val)
        np.save(f'../data/input_test_{stock}.npy', input_test)
        logging.info('\n---------- DONE ----------')
    else:
        logging.info('\nLoading train, validation and test sets...')
        input_train = np.load(f'../data/input_train_{stock}.npy')
        input_val = np.load(f'../data/input_val_{stock}.npy')
        input_test = np.load(f'../data/input_test_{stock}.npy')
        condition_train = np.load(f'../data/condition_train_{stock}.npy')
        condition_val = np.load(f'../data/condition_val_{stock}.npy')
        condition_test = np.load(f'../data/condition_test_{stock}.npy')


    # Define the parameters of the GAN.
    # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
    latent_dim = 100
    n_epochs = 1
    T_condition = condition_train.shape[1]
    T_real = input_train.shape[1]
    num_features = input_train.shape[2]
    n_units_generator = 100
    batch_size = 32
    gen_units = [5, 64, 64]
    disc_units = [gen_units[0], 64, 64]

    # Create a TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)

    conditioner_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)
    optimizer = [conditioner_optimizer, generator_optimizer, critic_optimizer]

    generator_model, condition_model = build_generator(T_condition, latent_dim, gen_units, T_real, num_features)
    critic_model = build_critic(T_real, T_condition, num_features, disc_units)
    # Define a dictionary to store the metrics
    metrics = {'critic_loss': [], 'gen_loss': [], 'real_acc': [], 'fake_acc': []}
    for epoch in range(n_epochs):
        i = 0
        for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
            i += 1
            train_step(batch_real_samples, batch_conditions, condition_model, generator_model, critic_model, optimizer, T_condition, latent_dim, i, metrics)