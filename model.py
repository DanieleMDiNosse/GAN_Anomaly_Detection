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
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy


def build_generator(T_cond, latent_dim, K, T_real, num_features):
    # ----------------- CONDITIONER -----------------
    condition_input = Input(shape=(T_cond, num_features))
    condition_lstm = LSTM(95, return_sequences=True)(condition_input)
    dense_layer = Dense(K)
    condition_output = TimeDistributed(dense_layer)(condition_lstm)
    condition_model = Model(condition_input, condition_output)
    condition_model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(condition_model, to_file=f'plots/condition_model_plot_{os.getpid()}.png', show_shapes=True, show_layer_names=True)

    # ----------------- GENERATOR -----------------
    generator_condition_input = Input(shape=(T_cond, K,))
    generator_noise_input = Input(shape=(T_cond, latent_dim,))
    generator_input = Concatenate()([generator_condition_input, generator_noise_input])
    generator_lstm = LSTM(10)(generator_input)
    generator_dense = Dense(num_features*T_real)(generator_lstm)
    generator_reshape = Reshape((T_real, num_features))(generator_dense)
    generator_model = Model([generator_condition_input, generator_noise_input], generator_reshape)

    generator_model.compile(optimizer='adam', loss='mean_squared_error')
    plot_model(generator_model, to_file=f'plots/generator_model_plot_{os.getpid()}.png', show_shapes=True, show_layer_names=True)
    return generator_model, condition_model

def build_discriminator(T_real, T_cond, num_features, K):
    # Conditioner input
    discriminator_condition_input = Input(shape=(T_cond, K,))
    discriminator_lstm = LSTM(95)(discriminator_condition_input)
    dense_layer = Dense(T_real*num_features)(discriminator_lstm)
    discriminator_reshape = Reshape((T_real, num_features))(dense_layer)

    discriminator_input = Input(shape=(T_real, num_features))
    discriminator_concat = Concatenate()([discriminator_reshape, discriminator_input])

    discriminator_lstm = LSTM(10)(discriminator_concat)
    discriminator_output = Dense(1, activation='sigmoid')(discriminator_lstm)

    discriminator_model = Model([discriminator_condition_input, discriminator_input], discriminator_output)
    discriminator_model.compile(loss='binary_crossentropy', optimizer='adam')
    # plot the models
    plot_model(discriminator_model, to_file=f'plots/discriminator_model_plot_{os.getpid()}.png', show_shapes=True, show_layer_names=True)

    return discriminator_model

def train_step(real_samples, conditions, condition_model, generator_model, discriminator_model, T_cond, latent_dim, i):
    # Initialize random noise
    noise = tf.random.normal([batch_conditions.shape[0], T_cond, latent_dim])

    # Create a GradientTape for the conditioner, generator, and discriminator.
    # GrandietTape collects all the operations that are executed inside it.
    # Then this operations are used to compute the gradeints of the loss function with 
    # respect to the trainable variables.

    # Discriminator training
    with tf.GradientTape(persistent=True) as tape:
        # Ensure the tape is watching the trainable variables of conditioner
        tape.watch(condition_model.trainable_variables)
        # Step 1: Use condition_model to preprocess conditions and get K values
        k_values = condition_model(conditions, training=True)
        # Step 2: Generate fake samples using generator
        generated_samples = generator_model([k_values, noise], training=True)
        # Step 3: Discriminator distinguishes real and fake samples
        real_output = discriminator_model([k_values, real_samples], training=True)
        fake_output = discriminator_model([k_values, generated_samples], training=True)
        # Compute the losses
        disc_loss = compute_discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = tape.gradient(disc_loss, discriminator_model.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    # Generator training
    noise = tf.random.normal([batch_real_samples.shape[0], T_cond, latent_dim])
    with tf.GradientTape(persistent=True) as tape:
        # Ensure the tape is watching the trainable variables of conditioner
        tape.watch(condition_model.trainable_variables)
        # Step 1: Use condition_model to preprocess conditions and get K values
        k_values = condition_model(conditions, training=True)
        # Step 2: Generate fake samples using generator
        generated_samples = generator_model([k_values, noise], training=True)
        # Step 3: Discriminator distinguishes real and fake samples
        fake_output = discriminator_model([k_values, generated_samples], training=True)
        # Compute the losses
        gen_loss = compute_generator_loss(fake_output)

    # Calculate gradients
    gradients_of_generator = tape.gradient(gen_loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    
    # Apply gradients to update weights
    gradients_of_conditioner = tape.gradient([gen_loss, disc_loss], condition_model.trainable_variables)
    conditioner_optimizer.apply_gradients(zip(gradients_of_conditioner, condition_model.trainable_variables))
    
    # Delete the tape to free resources
    del tape

    if i % 10 == 0:
        summarize_performance(real_output, fake_output, disc_loss, gen_loss, generated_samples)

    return None

def compute_discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + fake_loss
    return tf.reduce_mean(total_disc_loss)

def compute_generator_loss(fake_output):
    return tf.reduce_mean(binary_crossentropy(tf.ones_like(fake_output), fake_output))

def summarize_performance(real_output, fake_output, disc_loss, gen_loss, generated_samples):
    # Calculate Discriminator Accuracy on Real Samples
    real_labels = np.ones((real_output.shape[0], 1))  # Real samples have label 1
    real_accuracy = accuracy_score(real_labels, np.round(real_output.numpy()))
    
    # Calculate Discriminator Accuracy on Fake Samples
    fake_labels = np.zeros((fake_output.shape[0], 1))  # Fake samples have label 0
    fake_accuracy = accuracy_score(fake_labels, np.round(fake_output.numpy()))

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(disc_loss, label='Discriminator loss')
    plt.plot(gen_loss, label='Generator loss')
    plt.xlabel('Batch x 10')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/losses_epoch_{os.getpid()}.png')

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(real_accuracy, label='Accuracy real')
    plt.plot(fake_accuracy, label='Accuracy fake')
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

    print(f'Batch: {i} | Disc loss: {disc_loss:.5f} | Gen loss: {gen_loss:.5f} | Real acc: {real_accuracy:.5f} | Fake acc: {fake_accuracy:.5f}')
    return None

'''def train_GAN(generator, discriminator, conditioner, gan,  dataset, latent_dim, n_epochs, window_size, batch_size, scaler):
    # # How many windows I use for each epoch.
    # dataset.shape[0] is the number of windows in the dataset (e.g. in the train set).
    batch_per_epoch = int(dataset.shape[0] / batch_size)
    d_loss_real = []
    d_loss_fake = []
    g_loss = []
    c_loss = []
    acc_real = []
    acc_fake = []
    for i in tqdm(range(n_epochs), desc='Epochs'):
        for j in range(batch_per_epoch):
            # Prepare the points in the latent space as input for the generator
            z = random_inputs(batch_size, window_size, latent_dim)
            # Prepare the output of the conditioner as an additional input for the generator
            #cl = co
            # print('Dimension of the random sample: ', z.shape)
            # Generate fake samples
            X_fake = generator.predict(z)
            # print('Dimension of the fake sample: ', X_fake.shape)
            # Select a batch of random real samples
            idx = np.random.randint(0, dataset.shape[0], batch_size)
            # Retrieve real images
            X_real = dataset[idx]
            # print('Dimension of the real sample: ', X_real.shape)
            # Create the labels for the discriminator
            y_real = np.ones((batch_size, 1))
            y_fake = np.zeros((batch_size, 1))
            # Train the discriminator
            dlr, _ = discriminator.train_on_batch(X_real, y_real)
            dlf, _ = discriminator.train_on_batch(X_fake, y_fake)
            d_loss_real.append(dlr)
            d_loss_fake.append(dlf)
            # Plot d_loss_real and d_loss_fake on the first axis of the figure
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
            axes[0].plot(d_loss_real, label='d_loss_real', alpha=0.5)
            axes[0].plot(d_loss_fake, label='d_loss_fake', alpha=0.5)
            axes[0].plot(np.add(d_loss_real, d_loss_fake)*0.5, label='d_loss')
            axes[0].set_title('Discriminator loss')
            axes[0].set_xlabel('Batch')
            axes[0].set_ylabel('Loss')
            # Prepare the points in the latent space as input for the generator
            z = random_inputs(batch_size, window_size, latent_dim)
            # Create inverted labels for the fake samples
            y_gan = np.ones((batch_size, 1))
            # Update the generator via the discriminator's error
            gl = gan.train_on_batch(z, y_gan)
            g_loss.append(gl)
            # Plot the g_loss on the second axis of the figure
            axes[1].plot(g_loss, label='g_loss')
            axes[1].set_title('Generator loss')
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Loss')
            axes[0].legend()
            axes[1].legend()
            plt.savefig(f'plots/losses_epoch_{i}_{os.getpid()}.png')
            plt.close()
            # summarize loss on this batch
            print(f'Batch: {j/batch_per_epoch*100:.2f}%, d_loss (fake, real): {dlf:.5f}, {dlr:.5f}, gan_loss: {gl:.5f}')
            if j % 10 == 0:
                ar, af = summarize_performance(i, generator, discriminator, dataset, latent_dim, scaler, n_samples=100)
                acc_real.append(ar)
                acc_fake.append(af)
                # Plot the accurancy of the discriminator on the real and fake samples
                plt.figure(figsize=(10, 5), tight_layout=True)
                plt.plot(acc_real, label='Accuracy real')
                plt.plot(acc_fake, label='Accuracy fake')
                plt.title('Accuracy of the discriminator')
                plt.xlabel('Batch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.savefig(f'plots/accuracy_epoch_{i}_{os.getpid()}.png')
                plt.close()
    return None'''

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
    K = 5
    T_real = input_train.shape[1]
    num_features = input_train.shape[2]
    n_units_generator = 100
    batch_size = 32

    # Create a TensorFlow Dataset object
    dataset = tf.data.Dataset.from_tensor_slices((input_train, condition_train)).batch(batch_size)
    
    conditioner_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator_model, condition_model = build_generator(T_condition, latent_dim, K, T_real, num_features)
    discriminator_model = build_discriminator(T_real, T_condition, num_features, K)
    for epoch in range(n_epochs):
        i = 0
        for batch_real_samples, batch_conditions in tqdm(dataset, desc=f'Epoch {epoch+1}/{n_epochs}'):
            i += 1
            train_step(batch_real_samples, batch_conditions, condition_model, generator_model, discriminator_model, T_condition, latent_dim, i)


    # # Build the models
    # logging.info('\n---------- BUILDING THE MODELS ----------')
    # # generator = build_generator(input_shape_conditioner, num_features_output_cond, input_shape_unconditional, num_features, n_units_generator)
    # # logging.info('\nGenerator summary:')
    # # generator.summary()
    # # plot_model(generator, to_file='plots/generator_plot.png', show_shapes=True, show_layer_names=True)
    # # exit()
    # discriminator = build_discriminator((window_size, num_features), n_units_discriminator)
    # logging.info('\nDiscriminator summary:')
    # plot_model(discriminator, to_file='plots/discriminator_plot.png', show_shapes=True, show_layer_names=True)
    # discriminator.summary()
    # exit()
    # gan = build_gan(generator, discriminator)
    # logging.info('\nGAN summary:')
    # gan.summary()
    # plot_model(gan, to_file='plots/gan_plot.png', show_shapes=True, show_layer_names=True)
    # logigng.info('\n---------- CGAN TRAINING ----------')
    # train_GAN(generator, discriminator, gan, train, latent_dim, n_epochs, window_size, batch_size, scaler)
    





