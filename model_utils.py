'''This script contains the functions used to contruct and train the GAN.'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import psutil
import gc
import time
import subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import tensorflow as tf
from data_utils import *
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance
import argparse
# from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras.models import Model

def free_memory():
    tf.keras.backend.clear_session()
    gc.collect()

def log_memory_usage():
    process = psutil.Process(os.getpid())
    logging.info(f'RAM Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB')

def log_gpu_memory():
    '''Function to get the GPU memory usage. It returns the free memory in MB.'''
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], capture_output=True, text=True)
    logging.info(f'Free VRAM: {result.stdout.strip()} MB')
    return result.stdout.strip()

def conv_block(xi, filters, kernel_size, strides, padding, skip_connections):
    x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(xi)
    # x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    xo = layers.Dropout(0.2)(x)
    if skip_connections == True:
        x = layers.Concatenate()([xi, xo])
    return x

def lstm_block(xi, units, skip_connections):
    x = layers.LSTM(units=units, return_sequences=True)(xi)
    x = layers.Dropout(0.2)(x)
    # xo = layers.BatchNormalization()(x)
    if skip_connections == True:
        x = layers.Concatenate()([xi, x])
    return x

def dense_block(xi, units, skip_connections):
    x = layers.Dense(units=units)(xi)
    # xo = layers.LeakyReLU()(x)
    x = layers.ReLU()(x)
    # xo = layers.BatchNormalization()(xo)
    x = layers.Dropout(0.2)(x)
    if skip_connections == True:
        x = layers.Concatenate()([xi, x])
    return x

def build_discriminator(n_layers, type, skip_connections, T_gen, T_condition, num_features_input, num_features_gen, activate_condition=False, loss='original'):
    '''Build the discriminator model'''
    n_nodes = [2**(5+i) for i in range(n_layers)][::-1]

    if activate_condition == True:
        condition = layers.Input(shape=(T_condition, num_features_input), name='condition')
        if type == 'dense':
            x_c = layers.Flatten()(condition)
            for i in range(n_layers):
                x_c = dense_block(x_c, units=n_nodes[i], skip_connections=skip_connections)
        if type == 'conv':
            x_c = layers.Reshape((T_condition, T_condition, num_features_input))(condition)
            for i in range(n_layers):
                x_c = conv_block(x_c, filters=n_nodes[i], kernel_size=(3,num_features_input), strides=1, padding='same', skip_connections=skip_connections)
        if type == 'lstm':
            x_c = condition
            for i in range(n_layers):
                x_c = lstm_block(x_c, units=n_nodes[i], skip_connections=skip_connections)
            x_c = layers.Flatten()(x_c)
    else:
        T_gen = T_gen# + T_condition

    input = layers.Input(shape=(T_gen, num_features_gen), name='input')

    if type == 'conv':
        x = layers.Reshape((T_gen, num_features_gen, 1))(input)
        for i in range(n_layers):
            x = conv_block(x, filters=n_nodes[i], kernel_size=(3,num_features_gen), strides=1, padding='same', skip_connections=skip_connections)

    if type == 'lstm':
        x = input
        for i in range(n_layers):
            x = lstm_block(x, units=n_nodes[i], skip_connections=skip_connections)

    if type == 'dense':
        x = layers.Flatten()(input)
        for i in range(n_layers):
            x = dense_block(x, units=n_nodes[i], skip_connections=skip_connections)

    x = layers.Flatten()(x)
    
    if activate_condition == True:
        x = layers.Concatenate()([x, x_c])
    
    x = layers.Dense(32)(x)

    if loss == 'original' or loss == 'original_fm':
        output = layers.Dense(1, activation='sigmoid')(x)
    elif loss == 'wasserstein':
        output = layers.Dense(1)(x)
    
    if activate_condition == True:
        discriminator = tf.keras.Model([input, condition], output, name='discriminator')
    else:
        discriminator = tf.keras.Model([input], output, name='discriminator')
    
    # plot_model(discriminator, to_file='models/discriminator_plot.png', show_shapes=True, show_layer_names=True)
    return discriminator

def build_generator(n_layers, type, skip_connections, T_gen, T_condition, num_features_input, num_features_gen, latent_dim, activate_condition=False):
    '''Build the generator model'''
    n_nodes = [2**(5+i) for i in range(n_layers)][::-1]
    # n_nodes = [64, 48, 32]

    if activate_condition == True:
        condition = layers.Input(shape=(T_condition, num_features_input), name='condition')
        if type == 'dense':
            x_c = layers.Flatten()(condition)
            for i in range(n_layers):
                x_c = dense_block(x_c, units=n_nodes[i], skip_connections=skip_connections)
        if type == 'conv':
            x_c = layers.Reshape((T_condition, T_condition, num_features_input))(condition)
            for i in range(n_layers):
                x_c = conv_block(x_c, filters=n_nodes[i], kernel_size=(3,num_features_input), strides=1, padding='same', skip_connections=skip_connections)
        if type == 'lstm':
            x_c = condition
            for i in range(n_layers):
                x_c = lstm_block(x_c, units=n_nodes[i], skip_connections=skip_connections)
            x_c = layers.Flatten()(x_c)
    else:
        T_gen = T_gen# + T_condition

    input = layers.Input(shape=(T_gen*latent_dim, num_features_gen), name='input')

    if type == 'conv':
        x = layers.Reshape((T_gen, latent_dim,  num_features_gen))(input)
        for i in range(n_layers):
            x = conv_block(x, filters=n_nodes[i], kernel_size=(3,num_features_input), strides=1, padding='same', skip_connections=skip_connections)

    if type == 'lstm':
        x = input
        for i in range(n_layers):
            x = lstm_block(x, units=n_nodes[i], skip_connections=skip_connections)
 
    if type == 'dense':
        x = layers.Flatten()(input)
        for i in range(n_layers):
            x = dense_block(x, units=n_nodes[i], skip_connections=skip_connections)

    xi = layers.Flatten()(x)

    if activate_condition == True:
        xi = layers.Concatenate()([xi, x_c])
    
    x = layers.Dense(32)(xi)

    x = layers.Dense(T_gen*num_features_gen, activation='linear')(x)

    output = layers.Reshape((T_gen, num_features_gen))(x)

    if activate_condition == True:
        generator_model = Model([input, condition], output, name='generator_model')
    else:
        generator_model = Model([input], output, name='generator_model')
    
    # plot_model(generator_model, to_file='models/generator_plot.png', show_shapes=True, show_layer_names=True)
    return generator_model

# @tf.function
def train_step(real_samples, condition, generator_model, noise, discriminator_model, feature_extractor, optimizer, loss, batch_size, clipping):
    discriminator_optimizer = optimizer[0]
    generator_optimizer = optimizer[1]

    if loss == 'original' or loss == 'original_fm':
        disc_step = 1
    elif loss == 'wasserstein':
        disc_step = 5

    '''Create a GradientTape for the generator and the discriminator.
    GrandietTape collects all the operations that are executed inside it.
    Then this operations are used to compute the gradients of the loss function with 
    respect to the trainable variables. Remember that 'persistent=True' is needed
    iff you want to compute the gradients of the operations inside the tape more than once
    (for example wrt different losses)

    Steps:
    Step 1: Generate fake samples using generator
    Step 2: discriminator distinguishes real and fake samples
    Step 3: Compute the losses
    Step 4: Compute the gradients of the losses wrt the trainable variables
    Step 5: Apply the gradients to the optimizer'''

    # Discriminator training
    for _ in range(disc_step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = generator_model([noise, condition], training=True)
            real_output = discriminator_model([real_samples, condition], training=True)
            fake_output = discriminator_model([generated_samples, condition], training=True)

            if loss == 'original' or loss == 'original_fm':
                discriminator_loss = compute_discriminator_loss(real_output, fake_output)
            elif loss == 'wasserstein':
                # The critic in a WGAN is trained to output higher scores for real data and lower scores for generated data
                discriminator_loss = wasserstein_loss([real_output, fake_output])
                gp = gradient_penalty(discriminator_model, batch_size, real_samples, generated_samples, condition)
                discriminator_loss += 10.0 * gp

        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator_model.trainable_variables)
        if clipping == True:
        # Clip the gradients to stabilize training
            gradients_of_discriminator = [tf.clip_by_norm(g, clip_norm=1) if g is not None else None for g in gradients_of_discriminator]
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    # Generator training
    with tf.GradientTape() as gen_tape:
        generated_samples = generator_model([noise, condition], training=True)
        fake_output = discriminator_model([generated_samples, condition], training=True)
        real_features_list = feature_extractor([real_samples, condition])
        generated_features_list = feature_extractor([generated_samples, condition])

        if loss == 'original':
            generator_loss = compute_generator_loss(fake_output)
        elif loss == 'original_fm':
            c = 1
            for r_sample in real_samples:
                if tf.reduce_sum(tf.sign(r_sample)) != 0:
                    c += 1
            generator_loss = compute_generator_loss(fake_output)
            fm_loss = compute_feature_matching_loss(real_features_list, generated_features_list)
            generator_loss += fm_loss + 0.5 * c
        elif loss == 'wasserstein':
            generator_loss = wasserstein_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(generator_loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    # Delete the tape to free resources
    del gen_tape
    del disc_tape

    return generated_samples, real_output, fake_output, discriminator_loss, generator_loss

def train_step_unconditional(real_samples, generator_model, noise, discriminator_model, feature_extractor, optimizer, loss, batch_size, clipping):
    discriminator_optimizer = optimizer[0]
    generator_optimizer = optimizer[1]

    if loss == 'original' or loss == 'original_fm':
        disc_step = 1
    elif loss == 'wasserstein':
        disc_step = 5

    # Discriminator training
    for _ in range(disc_step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = generator_model(noise, training=True)
            real_output = discriminator_model(real_samples, training=True)
            fake_output = discriminator_model(generated_samples, training=True)

            if loss == 'original' or loss == 'original_fm':
                discriminator_loss = compute_discriminator_loss(real_output, fake_output)
            # elif loss == 'wasserstein':
            #     # The critic in a WGAN is trained to output higher scores for real data and lower scores for generated data
            #     discriminator_loss = wasserstein_loss([real_output, fake_output])
            #     gp = gradient_penalty(discriminator_model, batch_size, real_samples, generated_samples)
            #     discriminator_loss += 10.0 * gp

        gradients_of_discriminator = disc_tape.gradient(discriminator_loss, discriminator_model.trainable_variables)
        if clipping == True:
        # Clip the gradients to stabilize training
            gradients_of_discriminator = [tf.clip_by_norm(g, clip_norm=1) if g is not None else None for g in gradients_of_discriminator]
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

    # Generator training
    with tf.GradientTape() as gen_tape:
        generated_samples = generator_model(noise, training=True)
        fake_output = discriminator_model(generated_samples , training=True)
        real_features_list = feature_extractor(real_samples )
        generated_features_list = feature_extractor(generated_samples)

        if loss == 'original':
            generator_loss = compute_generator_loss(fake_output)
        elif loss == 'original_fm':
            generator_loss = compute_generator_loss(fake_output)
            fm_loss = compute_feature_matching_loss(real_features_list, generated_features_list)
            generator_loss += fm_loss
        elif loss == 'wasserstein':
            generator_loss = wasserstein_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(generator_loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))

    # Delete the tape to free resources
    del gen_tape
    del disc_tape

    return generated_samples, real_output, fake_output, discriminator_loss, generator_loss

def build_feature_extractor(discriminator, layer_indices):
    '''Build a feature extractor model from the discriminator model. This model will be used to compute the feature matching loss.'''
    outputs = [discriminator.layers[i].output for i in layer_indices]
    for output in outputs:
        logging.info(f'Feature Extractor: Extracted feature from layer with shape: {output.shape}')
    return tf.keras.Model(discriminator.input, outputs)

# @tf.function
def compute_feature_matching_loss(real_features_list, generated_features_list):
    losses = [tf.reduce_mean(tf.square(real - generated))
              for real, generated in zip(real_features_list, generated_features_list)]
    return tf.reduce_mean(losses)  # Average over all the feature matching losses

# @tf.function
def compute_discriminator_loss(real_output, fake_output):
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss

# @tf.function
def compute_generator_loss(fake_output):
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    # return -tf.reduce_mean(tf.math.log(fake_output + 1e-5))
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)

# @tf.function
def wasserstein_loss(predictions):
    if len(predictions) == 2:
        real_output, fake_output = predictions
        w_real = tf.reduce_mean(real_output)
        w_fake = tf.reduce_mean(fake_output)
        w_tot = w_fake - w_real
    else:
        fake_output = predictions
        w_tot = -tf.reduce_mean(fake_output)
    return w_tot

# @tf.function
def gradient_penalty(discriminator_model, batch_size, real_samples, fake_samples, condition):
    # Get the interpolated samples
    alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
    real_samples = tf.cast(real_samples, tf.float32)
    diff = fake_samples - real_samples
    interpolated = real_samples + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator_model([interpolated, condition], training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return float(gp)

def summarize_performance(real_output, fake_output, discriminator_loss, gen_loss, metrics, job_id, args):

    real_output = real_output.numpy()
    fake_output = fake_output.numpy()
    # features = [f'Curve{i+1}' for i in range(generated_samples.shape[1])]

    # add the metrics to the dictionary
    metrics['gen_loss'].append(np.mean(gen_loss))
    metrics['real_disc_out'].append(np.mean(real_output))
    metrics['fake_disc_out'].append(np.mean(fake_output))
    metrics['discriminator_loss'].append(np.mean(discriminator_loss))

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['discriminator_loss'], label='Discriminator loss', alpha=0.7)
    plt.plot(metrics['gen_loss'], label='Generator loss', alpha=0.7)
    plt.xlabel('Epoch x 10')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/0_losses.png')

    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.plot(metrics['real_disc_out'], label='Real', alpha=0.7)
    plt.plot(metrics['fake_disc_out'], label='Fake', alpha=0.7)
    plt.xlabel('Epoch x 10')
    plt.ylabel('Discriminator output')
    plt.legend()
    plt.savefig(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/1_disc_output.png')

    plt.close('all')
    return None

def overall_wasserstein_distance(generator_model, dataset_train, noise, conditional):
    gen_samples = []
    real_samples = []
    
    j = 0
    # Generate all the samples
    if conditional == True:
        for batch_condition, batch in dataset_train:
            batch_size = batch_condition.shape[0]
            gen_sample = generator_model([noise[j*batch_size:(j+1)*batch_size], batch_condition])
            j += 1
            for i in range(gen_sample.shape[0]):
                # All the appended samples will be of shape (T_gen, n_features_gen)
                gen_samples.append(gen_sample[i, -1, :].numpy())
                real_samples.append(batch_condition[i, -1, :].numpy())
    else:
        for batch in dataset_train:
            batch_size = batch[0].shape[0]
            gen_sample = generator_model(noise[j*batch_size:(j+1)*batch_size])
            j += 1
            for i in range(gen_sample.shape[0]):
                # All the appended samples will be of shape (T_gen, n_features_gen)
                gen_samples.append(gen_sample[i, -1, :].numpy())
                real_samples.append(batch[i, -1, :].numpy())

    gen_samples = np.array(gen_samples)
    real_samples = np.array(real_samples)
    n_features_gen = batch.shape[2]
    W_features = []
    for feature in range(n_features_gen):
        w = wasserstein_distance(real_samples[:, feature], gen_samples[:, feature])
        W_features.append(w)
    W_features = np.array(W_features)
    overall_W_mean = np.mean(np.array(W_features))

    return overall_W_mean, gen_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script contains several functions for building and training a GAN. It has been used also as test script for the GAN on some synthetic data.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument('-d', '--data', type=str, help='Which data to use (sine, step)')
    parser.add_argument('-nd', '--num_dim', help='Number of dimensions of the data', default=1, type=int)
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of hidden layers in the generator', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of hidden layers in the discriminator', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-Tc', '--T_condition', help='Number of time steps to condition on', type=int, default=1)
    parser.add_argument('-Tg', '--T_gen', help='Number of time steps to generate', type=int, default=1)
    parser.add_argument('-ls', '--loss', help='Loss function (original, wasserstein)', type=str, default='original')

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

    logging.basicConfig(filename=f'output_{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}.log', format='%(message)s', level=levels[args.log])

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
    os.mkdir(f'plots/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Model architecture plots, metrics plots
    os.mkdir(f'generated_samples/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Generated samples
    os.mkdir(f'models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}') # Models

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
    window_size = 3
    T_condition = args.T_condition
    T_gen = window_size - T_condition
    n_features_input = input_train.shape[1]
    latent_dim = 10
    n_epochs = 2000
    batch_size = 32

    windows = np.array(divide_into_windows(input_train, window_size))
    #Scale the data along the last axis using StandardScaler
    scaler = StandardScaler()
    # windows = scaler.fit_transform(windows.reshape(windows.shape[0], windows.shape[1]*windows.shape[2])).reshape(windows.shape)
    windows = scaler.fit_transform(windows.reshape(-1, windows.shape[-1])).reshape(windows.shape)
    logging.info(f'Windows shape:\n\t{windows.shape}')

    logging.info('\nSplit the condition data into train, validation sets...')
    train, val = windows[:int(windows.shape[0]*0.75)], windows[int(windows.shape[0]*0.75):]
    logging.info(f'Train shape:\n\t{train.shape}\nValidation shape:\n\t{val.shape}')
    logging.info('Done.')

    logging.info('\nDividing each window into condition and input...')
    condition_train, input_train = train[:, :T_condition, :], train[:, T_condition:, :]
    condition_val, input_val = val[:, :T_condition, :], val[:, T_condition:, :]
    logging.info('Done.')

    # Use logging.info to logging.info all the hyperparameters
    logging.info(f'\nHYPERPARAMETERS:\n\tlatent_dim per time: {latent_dim}\n\tn_features: {n_features_input}\n\tn_epochs: {n_epochs}\n\tT_condition: {T_condition}\n\tT_gen: {T_gen}\n\tbatch_size: {batch_size} -> num_batches: {train.shape[0]//batch_size}\n\tloss: {args.loss}\n')

    # Define the optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    optimizer = [discriminator_optimizer, generator_optimizer]

    # Define the loss function
    binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    # Build the models
    generator_model = build_generator(args.n_layers_gen, args.type_gen, True, T_gen, T_condition, n_features_input, latent_dim, True)
    discriminator_model = build_discriminator(args.n_layers_disc, args.type_disc, True, T_gen, T_condition, n_features_input, True)
    feature_extractor = build_feature_extractor(discriminator_model, [i for i in range(1, args.n_layers_disc)])

    logging.info('\n[Model] ---------- MODEL SUMMARIES ----------')
    logging.info(f'Data used\n\t{args.data} {args.num_dim}D')
    logging.info(f'Loss function\n\t{args.loss}')
    logging.info(f'Generator model\n\tType: {args.type_gen}\n\tNumber of layers: {args.n_layers_gen}')
    logging.info(f'Discriminator model\n\tType: {args.type_disc}\n\tNumber of layers: {args.n_layers_disc}')
    generator_model.summary(print_fn=logging.info)
    logging.info('\n')
    discriminator_model.summary(print_fn=logging.info)
    logging.info('[Model] ---------- DONE ----------\n')

    # Define a dictionary to store the metrics
    metrics = {'discriminator_loss': [], 'gen_loss': [], 'real_disc_out': [], 'fake_disc_out': []}

    # Define checkpoint and checkpoint manager
    checkpoint_prefix = f"models/{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.T_condition}_{args.loss}/"
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                    generator_model=generator_model,
                                    discriminator_model=discriminator_model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_prefix, max_to_keep=3)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)


    # Train the GAN.
    logging.info('\n[Training] ---------- START TRAINING ----------')
    if args.condition == True:
        dataset_train = tf.data.Dataset.from_tensor_slices((condition_train, input_train)).batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices((condition_val, input_val)).batch(batch_size)
    else:
        dataset_train = tf.data.Dataset.from_tensor_slices((train)).batch(batch_size)
        dataset_val = tf.data.Dataset.from_tensor_slices((val)).batch(batch_size)

    num_batches = len(dataset_train)
    logging.info(f'Number of training batches:\n\t{num_batches}\n')

    best_gen_weights = None
    best_disc_weights = None
    best_wass_dist = float('inf')
    patience_counter = 0
    patience = 0  # set your patience

    for epoch in range(n_epochs):
        j = 0
        for batch_condition, batch_real_samples in dataset_train:
            j += 1
            batch_size = batch_real_samples.shape[0]
            generator_model, discriminator_model = train_step(batch_real_samples, batch_condition, generator_model, discriminator_model, feature_extractor, optimizer, args.loss, T_gen, T_condition, latent_dim, batch_size, num_batches, j, job_id, epoch, metrics, args)
            # if j == 10:
            #     break
        # Save the models via checkpoint
        checkpoint_manager.save()

        logging.info('Creating a time series with the generated samples...')
        number_of_batches_plot = 5
        features = [f'Curve{i+1}' for i in range(args.num_dim)]
        # scale back the data
        plot_samples(dataset_train, number_of_batches_plot, generator_model, features, T_gen, T_condition, latent_dim, n_features_input, job_id, epoch, scaler, args, final=False)
        logging.info('Done')

