'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import divide_into_windows
import argparse
import logging
from sklearn.model_selection import train_test_split
import os

# Generate a GAN in which both the discriminator and the generator are LSTM networks with 100 units and 3 layers
def build_generator(input_shape, output_shape, n_units):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(n_units, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(n_units, return_sequences=True),
        tf.keras.layers.LSTM(n_units, return_sequences=True),
        tf.keras.layers.Dense(np.prod(output_shape), activation='tanh'),
        tf.keras.layers.Reshape(output_shape)
    ])
    return model

def build_discriminator(input_shape, n_units):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(n_units, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(n_units, return_sequences=True),
        tf.keras.layers.LSTM(n_units, return_sequences=True),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # compile the model adding as loss the binary crossentropy and as optimizer the Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = tf.keras.Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def random_inputs(batch_size, latent_dim):
    return np.random.randn(batch_size, latent_dim)

def train_GAN(generator, discriminator, gan,  dataset, latent_dim, n_epochs, n_batches):
    # Evaluate the number of batches per epoch
    batch_per_epoch = int(dataset.shape[0] / n_batches)
    # Calculate the number of training iterations
    n_steps = batch_per_epoch * n_epochs
    for i in range(n_steps):

        # Prepare the points in the latent space as input for the generator
        z = random_inputs(n_batches, latent_dim)
        # Generate fake samples
        X_fake = generator.predict(z)
        # Select a batch of random real images
        idx = np.random.randint(0, dataset.shape[0], n_batches)
        # Retrieve real images
        X_real = dataset[idx]
        # Create the labels for the discriminator
        y_real = np.ones((n_batches, 1))
        y_fake = np.zeros((n_batches, 1))
        # Train the discriminator
        discriminator.train_on_batch(X_real, y_real)
        discriminator.train_on_batch(X_fake, y_fake)

        # Prepare the points in the latent space as input for the generator
        z = random_inputs(n_batches, latent_dim)
        # Create inverted labels for the fake samples
        y_gan = np.ones((n_batches, 1))
        # Update the generator via the discriminator's error
        gan.train_on_batch(z, y_gan)

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
    logging.basicConfig(level=levels[args.log])

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


    # Divide the dataframes into windows. This will be done for each day (each dataframe in dataframes).
    # For now I will focus just on the first day as a proof of concept.
    window_size = 500
    data = dataframes[0].values
    input_data = np.array(divide_into_windows(data, window_size))
    
    # Divide the data into train, validation and test set
    train, test = train_test_split(input_data, test_size=0.2, shuffle=False)
    train, val = train_test_split(train, test_size=0.2, shuffle=False)


    # Define the parameters of the GAN.
    # Remember that the number of samples for each batch is equal to the number of windows in the dataset divided by the number of batches.
    latent_dim = 100
    n_epochs = 1
    n_batches = 256

    # Train the GAN. The input shape for the generator is equal to the shape of dimension
    # of the latent space, while the output shape is equal to the shape of the windows.
    # Conversely, the input shape for the discriminator is equal to the shape of the windows,
    # while the output shape is equal to 1 (since the discriminator is a binary classifier).
    generator = build_generator((latent_dim, 1), (window_size, 10), 100)
    discriminator = build_discriminator((window_size, 10), 100)
    gan = build_gan(generator, discriminator)
    train_GAN(generator, discriminator, gan, train, latent_dim, n_epochs, n_batches)
    





