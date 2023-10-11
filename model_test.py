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
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tcn import TCN
from scipy.stats import wasserstein_distance

def conv_block(xi, filters, kernel_size, strides, padding, skip_connections):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(xi)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    xo = layers.Dropout(0.2)(x)
    if skip_connections == True:
        x = layers.Concatenate()([xi, xo])
    return x

def lstm_block(xi, units, skip_connections):
    x = layers.LSTM(units=units, return_sequences=True)(xi)
    x = layers.Dropout(0.2)(x)
    xo = layers.LSTM(units=units, return_sequences=True)(x)
    if skip_connections == True:
        x = layers.Concatenate()([xi, xo])
    return x

def dense_block(xi, units, skip_connections):
    x = layers.Dense(units=units)(xi)
    x = layers.LeakyReLU()(x)
    xo = layers.Dropout(0.2)(x)
    if skip_connections == True:
        x = layers.Concatenate()([xi, xo])
    return x

def build_discriminator1(n_layers, type, skip_connections, window_size, num_features_input):
    input = layers.Input(shape=(window_size, num_features_input), name='input')

    if type == 'conv':
        x = layers.Reshape((window_size, num_features_input, 1))(input)
        for _ in range(n_layers):
            x = conv_block(x, filters=50, kernel_size=(5,1), strides=1, padding='same', skip_connections=skip_connections)

    if type == 'lstm':
        for _ in range(n_layers):
            x = input
            x = lstm_block(x, units=50, skip_connections=skip_connections)

    if type == 'dense':
        x = layers.Flatten()(input)
        for _ in range(n_layers):
            x = dense_block(x, units=100, skip_connections=skip_connections)

    xi = layers.Flatten()(x)
    x = layers.Dense(50)(xi)
    if skip_connections == True:
        x = layers.Concatenate()([xi, x])
    output = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = tf.keras.Model([input], output, name='discriminator')
    return discriminator

def build_generator1(n_layers, type, skip_connections, window_size, num_features_input):
    input = layers.Input(shape=(window_size, num_features_input), name='input')

    if type == 'conv':
        x = layers.Reshape((window_size, num_features_input, 1))(input)
        for _ in range(n_layers):
            x = conv_block(x, filters=50, kernel_size=(5,1), strides=1, padding='same', skip_connections=skip_connections)

    if type == 'lstm':
        for _ in range(n_layers):
            x = input
            x = lstm_block(x, units=50, skip_connections=skip_connections)
 
    if type == 'dense':
        x = layers.Flatten()(input)
        for _ in range(n_layers):
            x = dense_block(x, units=100, skip_connections=skip_connections)

    xi = layers.Flatten()(x)
    x = layers.Dense(window_size*num_features_input)(xi)
    x = layers.LeakyReLU()(x)

    output = layers.Reshape((window_size, num_features_input))(x)
    generator_model = Model([input], output, name='generator_model')
    return generator_model

def build_discriminator(window_size, num_features_input):
    input = layers.Input(shape=(window_size, num_features_input), name='input')
    x = layers.Reshape((window_size, num_features_input, 1))(input)

    x = layers.Conv2D(filters=50, kernel_size=(5,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2D(filters=25, kernel_size=(5,1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(50)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    discriminator = tf.keras.Model([input], output, name='discriminator')
    return discriminator

def build_generator(latent_dim, window_size, num_features_input):
    input = layers.Input(shape=(window_size, num_features_input), name='input')

    reshape = layers.Reshape((window_size, num_features_input, 1))(input)
    conv2d_1 = layers.Conv2D(filters=32, kernel_size=(5,1), padding='same')(reshape)
    x = layers.BatchNormalization()(conv2d_1)
    conv2d_1 = layers.LeakyReLU()(x)
    # add skip connection between input and conv2d_1
    concat_1 = layers.Concatenate()([reshape, conv2d_1])


    conv2d_2 = layers.Conv2D(filters=16, kernel_size=(5,1), padding='same')(concat_1)
    x = layers.BatchNormalization()(conv2d_2)
    conv2d_2 = layers.LeakyReLU()(x)
    # add skip connection between input and conv2d_2
    concat_2 = layers.Concatenate()([concat_1, conv2d_2])

    x = layers.Flatten()(concat_2)
    x = layers.Dense(window_size*num_features_input)(x)
    x = layers.LeakyReLU()(x)


    output = layers.Reshape((window_size, num_features_input))(x)
    generator_model = Model([input], output, name='generator_model')
    return generator_model

def train_step(real_samples, generator_model, discriminator_model, optimizer, window_size, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, disc_accuracy):
    discriminator_optimizer = optimizer[0]
    generator_optimizer = optimizer[1]

    noise = tf.random.normal([batch_size, window_size, real_samples.shape[2]])
    # real_samples = tf.cast(real_samples, tf.float32)
    # real_samples += tf.random.normal(real_samples.shape, mean=0.0, stddev=0.1)
    if disc_accuracy < 1.75:
        with tf.GradientTape() as disc_tape:
            generated_samples = generator_model(noise, training=True)
            real_output = discriminator_model(real_samples, training=True)
            fake_output = discriminator_model(generated_samples, training=True)
            discriminator_loss = compute_discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator_model.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))
    else:
        discriminator_loss = np.nan
        real_output = np.nan

    with tf.GradientTape() as gen_tape:
        generated_samples = generator_model(noise, training=True)
        fake_output = discriminator_model(generated_samples, training=True)
        generator_loss = compute_generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(generator_loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
        
    if disc_accuracy < 1.75:
        outputs = [real_output, fake_output]
        str = 'D trained'
    else:
        outputs = [fake_output]
        str = 'D not trained'

    disc_accuracy = compute_accuracy(outputs)
    if j % 10 == 0:
        logging.info(f'{str} | Epoch: {epoch} | Batch: {j}/{num_batches} | Disc loss: {np.mean(discriminator_loss):.5f} | Gen loss: {np.mean(generator_loss):.5f} | <Disc_output_r>: {np.mean(real_output):.5f}| <Disc_output_f>: {np.mean(fake_output):.5f} | Disc accuracy: {disc_accuracy:.3f}')
    if j % 50 == 0:
        summarize_performance(real_output, fake_output, discriminator_loss, generator_loss, generated_samples, real_samples, metrics, j, num_batches, job_id, epoch)

    return generator_model, discriminator_model, disc_accuracy

def compute_discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + fake_loss

    return total_disc_loss

def compute_generator_loss(fake_output):
    return -tf.math.log(fake_output + 1e-5)
    # return binary_crossentropy(tf.ones_like(fake_output), fake_output)

def summarize_performance(real_output, fake_output, discriminator_loss, gen_loss, generated_samples, real_samples, metrics, i, num_batches, job_id, epoch):

    generated_samples = generated_samples[0,:,:].numpy()
    real_samples = real_samples[0,:,:].numpy()
    features = [f'Curve{i+1}' for i in range(generated_samples.shape[1])]

    # add the metrics to the dictionary
    metrics['gen_loss'].append(np.mean(gen_loss))
    if np.isnan(real_output).any() == False:
        for score in real_output[1,:]:
            metrics['real_disc_out'].append(score)
        for score in fake_output[1,:]:
            metrics['fake_disc_out'].append(score)
        metrics['discriminator_loss'].append(np.mean(discriminator_loss))
    else:
        metrics['real_disc_out'].append(-0.2)
        metrics['fake_disc_out'].append(-0.2)
        metrics['discriminator_loss'].append(-0.2)

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['discriminator_loss'], label='Discriminator loss')
    plt.plot(metrics['gen_loss'], label='Generator loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_losses.png')

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_disc_out'], label='Real')
    plt.plot(metrics['fake_disc_out'], label='Fake')
    plt.xlabel('Batch')
    plt.ylabel('Discriminator output')
    plt.legend()
    plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_disc_output.png')

    if generated_samples.shape[1] == 1:
        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.plot(generated_samples)
        plt.xlabel('Time (Events)')
        plt.ylabel('Value')
        plt.title(f'Generated sample_{epoch}_{i}')
        plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/generated_samples_{epoch}_{i}.png')

        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.plot(real_samples, label='Real')
        plt.xlabel('Time (Events)')
        plt.ylabel('Value')
        plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_real_samples.png')
    else:
        # Plot a chosen generated sample
        fig, axes = plt.subplots(generated_samples.shape[1], 1, figsize=(10, 10), tight_layout=True)
        for j, feature in zip(range(generated_samples.shape[1]), features):
            axes[j].plot(generated_samples[:, j])
            axes[j].set_title(f'Generated {feature}_{epoch}_{i}')
        axes[j].set_xlabel('Time (Events)')
        plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/generated_samples_{epoch}_{i}.png')

        # Plot a chosen real sample
        fig, axes = plt.subplots(real_samples.shape[1], 1, figsize=(10, 10), tight_layout=True)
        for j, feature in zip(range(real_samples.shape[1]), features):
            axes[j].plot(real_samples[:, j])
            axes[j].set_title(f'Real {feature}')
        axes[j].set_xlabel('Time (Events)')
        plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_real_samples.png')

    # Create an animated gif of the generated samples
    if (i+50)/num_batches > 1:
        logging.info('Creating animated gif...')
        create_animated_gif(job_id, epoch, args.type_gen, args.type_disc, args.n_layers, args.data)

    plt.close('all')
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-d', '--data', type=str, help='Which data to use (sine, step)')
    parser.add_argument('-nd', '--num_dim', help='Number of dimensions of the data', default=1, type=int)
    parser.add_argument('-nl', '--n_layers', help='Type of model (conv, lstm, dense)', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)

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

    logging.basicConfig(filename=f'output_{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}.log', format='%(message)s', level=levels[args.log])

    logger = tf.get_logger()
    logger.setLevel('ERROR')

    # Set the seed for TensorFlow to the number of the beast
    tf.random.set_seed(666)

    current_datetime = pd.Timestamp.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f"Current Date and Time:\n\t {formatted_datetime}")

    # Enable device placement logging
    tf.debugging.set_log_device_placement(True)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) == 0:
        logging.info("No GPUs available.")
    else:
        logging.info("Available GPUs:")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        for device in physical_devices:
            logging.info(f'\t{device}\n')
    
    # Folders creation
    os.mkdir(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}') # Generated samples
    os.mkdir(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}') # Models

    # Generate syntehtic data
    if args.data == 'sine':
        input_train = np.array([[sin_wave(amplitude=i, omega=2*i, phi=np.random.uniform())] for i in range(1, args.num_dim+1)])
    elif args.data == 'step':
        input_train = np.array([[step_fun(i)] for i in range(1, args.num_dim+1)])
    elif args.data == 'ar1':
        input_train = np.array([[ar1()] for i in range(1, args.num_dim+1)])
    
    input_train = np.swapaxes(input_train, 2, 0)
    input_train = np.swapaxes(input_train, 1, 2)
    input_train = np.reshape(input_train, (input_train.shape[0], input_train.shape[1]))
    logging.info(f'Input shape:\n\t{input_train.shape}')

    # Define the parameters of the GAN.
    # Batch size: all the sample -> batch mode, one sample -> SGD, in between -> mini-batch SGD
    window_size = 64
    n_features_input = input_train.shape[1]
    latent_dim = 1
    n_epochs = 2000
    T_condition = int(window_size*0.5)
    T_real = window_size
    batch_size = 32

    windows = np.array(divide_into_windows(input_train, window_size))
    #Scale the data along the last axis using StandardScaler
    scaler = StandardScaler()
    windows = scaler.fit_transform(windows.reshape(windows.shape[0], windows.shape[1]*windows.shape[2])).reshape(windows.shape)
    logging.info(f'Windows shape:\n\t{windows.shape}')

    logging.info('\nSplit the condition data into train, validation sets...')
    train, val = windows[:int(windows.shape[0]*0.75)], windows[int(windows.shape[0]*0.75):]
    logging.info(f'Train shape:\n\t{train.shape}\nValidation shape:\n\t{val.shape}')
    logging.info('Done.')

    # logging.info('\nDividing each window into condition and input...')
    # condition_train, input_train = train[:, :T_condition, :], train[:, T_condition:, :]
    # condition_val, input_val = val[:, :T_condition, :], val[:, T_condition:, :]
    # condition_test, input_test = test[:, :T_condition, :], test[:, T_condition:, :]
    # logging.info('Done.')

    # Use logging.info to logging.info all the hyperparameters
    logging.info(f'\nHYPERPARAMETERS:\n\tlatent_dim: {latent_dim}\n\tn_features: {n_features_input}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_real: {T_real}\n\tbatch_size: {batch_size}\n\tnum_batches: {train.shape[0]//batch_size}')

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    optimizer = [discriminator_optimizer, generator_optimizer]

    # Define the loss function
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Build the models

    # generator_model = build_generator(latent_dim, window_size, n_features_input)
    # discriminator_model = build_discriminator(window_size, n_features_input)
    generator_model = build_generator1(args.n_layers, args.type_gen, True, window_size, n_features_input)
    discriminator_model = build_discriminator1(args.n_layers, args.type_disc, True, window_size, n_features_input)

    logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
    logging.info(f'Data used\n\t{args.data} {args.num_dim}D')
    generator_model.summary(print_fn=logging.info)
    logging.info('\n')
    discriminator_model.summary(print_fn=logging.info)
    logging.info('[Model] ---------- DONE ----------\n')

    # Define a dictionary to store the metrics
    metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

    # Define checkpoint and checkpoint manager
    checkpoint_prefix = f"models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    generator_model=generator_model,
                                    discriminator_model=discriminator_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=3)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)


    # Train the GAN.
    logging.info('\n[Training] ---------- START TRAINING ----------')
    dataset_train = tf.data.Dataset.from_tensor_slices((train)).batch(batch_size)
    dataset_val = tf.data.Dataset.from_tensor_slices((val)).batch(batch_size)
    num_batches = len(dataset_train)
    logging.info(f'Number of training batches:\n\t{num_batches}\n')

    disc_accuracy = 0
    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 12  # set your patience

    for epoch in range(n_epochs):
        j = 0
        for batch_real_samples in dataset_train:
            j += 1
            batch_size = batch_real_samples.shape[0]
            generator_model, discriminator_model, disc_acc = train_step(batch_real_samples, generator_model, discriminator_model, optimizer, window_size, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, disc_accuracy)
            disc_accuracy = disc_acc
        # Save the models via checkpoint
        checkpoint_manager.save()
    
        # Validation loop
        wass_dist = [[] for i in range(n_features_input)]
        logging.info('\n---------- VALIDATION ----------')
        for val_batch in dataset_val:  # assume val_data is your validation dataset
            noise = tf.random.normal([batch_size, window_size, n_features_input])
            generated_samples = generator_model(noise, training=True)
            for feature in range(generated_samples.shape[2]):
                for i in range(generated_samples.shape[0]):
                    w = wasserstein_distance(val_batch[i, :, feature], generated_samples[i, :, feature])
                    wass_dist[feature].append(w)

        # Plot the Wasserstein distances on different subplots and save the image
        if n_features_input == 1:
            plt.figure(figsize=(10, 5), tight_layout=True)
            plt.plot(wass_dist[0], alpha=0.7)
            plt.plot([np.mean(wass_dist[0]) for i in range(len(wass_dist[0]))], linestyle='--', color='red')
            plt.title(f'Wasserstein distance')
            plt.xlabel('Iterations')
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_wasserstein_{epoch}.png')
        else:
            fig, axes = plt.subplots(n_features_input, 1, figsize=(10, 10), tight_layout=True)
            for j, feature in zip(range(n_features_input), [f'Curve{i+1}' for i in range(n_features_input)]):
                axes[j].plot(wass_dist[j], alpha=0.7)
                axes[j].plot([np.mean(wass_dist[j]) for i in range(len(wass_dist[j]))], linestyle='--', color='red')
                axes[j].set_title(f'Wasserstein distance {feature}')
            axes[j].set_xlabel('Iterations')
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_wasserstein_{epoch}.png')

        wass_dist = np.mean(np.array(wass_dist).flatten())

        logging.info(f'Wasserstein distance: {wass_dist}')
        # Check early stopping criteria
        if wass_dist < best_wass_dist:
            best_wass_dist = wass_dist
            best_gen_weights = generator_model.get_weights()
            best_disc_weights = discriminator_model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logging.info(f"Early stopping on epoch {epoch}. Restoring best weights...")
            generator_model.set_weights(best_gen_weights)  # restore best weights
            discriminator_model.set_weights(best_disc_weights)
            # generate a sample and plot it
            noise = tf.random.normal([batch_size, window_size, n_features_input])
            generated_samples = generator_model(noise, training=True)
            generated_samples = generated_samples[0,:,:].numpy()
            features = [f'Curve{i+1}' for i in range(generated_samples.shape[1])]
            fig, axes = plt.subplots(generated_samples.shape[1], 1, figsize=(10, 10), tight_layout=True)
            for j, feature in zip(range(generated_samples.shape[1]), features):
                axes[j].plot(generated_samples[:, j])
                axes[j].set_title(f'Generated {feature}_{epoch}_{j}')
            axes[j].set_xlabel('Time (Events)')
            plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers}_{args.data}/000_final_generated_samples_{epoch}.png')
            logging.info('Done')
            break
        else:
            logging.info(f'Early stopping criterion not met. Patience counter:\n\t{patience_counter}')

