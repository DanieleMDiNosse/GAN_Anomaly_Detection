import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
# Change the environment variable TF_CPP_MIN_LOG_LEVEL to 2 to avoid the messages about the compilation of the CUDA code
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data_utils import explore_latent_space, explore_latent_space_loss
import argparse
import logging
from joblib import load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''This script is aimed to preprocess the data for model training. There is no much work to do: the idea is to simply feed the GAN with a multivariate time series
                        composed by a sliding window that shifts by one time step each time.''')
    parser.add_argument("-l", "--log", default="info",
                        help=("Provide logging level. Example --log debug', default='info'"))
    parser.add_argument("-j", "--job_id", type=str, help=("Provide job id"))
    parser.add_argument('-N', '--N_days', help='Number of days used for training', type=int, default=1)

    levels = {'critical': logging.CRITICAL,
              'error': logging.ERROR,
              'warning': logging.WARNING,
              'info': logging.INFO,
              'debug': logging.DEBUG}
    args = parser.parse_args()

    if os.getenv("PBS_JOBID") != None:
        job_id = os.getenv("PBS_JOBID")
    else:
        job_id = os.getpid()

    N = args.N_days

    # Initialize logger
    logging.basicConfig(filename=f'output_{job_id}_anomalydet.log', format='%(message)s', level=levels[args.log])
    
    paths = os.listdir('models')
    path = [p for p in paths if args.job_id in p]

    if len(path) == 0:
        raise ValueError('Job id not found')
    elif len(path) > 1:
        raise ValueError('Multiple job ids found')

    path = path[0]
    # model_path = [p for p in os.listdir(f'models/{path}') if 'h5' in p]
    gen_model_path = [p for p in os.listdir(f'models/{path}') if 'generator_model' in p]
    disc_model_path = [p for p in os.listdir(f'models/{path}') if 'discriminator_model' in p]
    # Load the models
    generator = tf.keras.models.load_model(f'models/{path}/{gen_model_path[0]}')
    discriminator = tf.keras.models.load_model(f'models/{path}/{disc_model_path[0]}')
    # print summary
    logging.info('Generator summary:')
    generator.summary(print_fn=lambda x: logging.info(x))
    logging.info('Discriminator summary:')
    discriminator.summary(print_fn=lambda x: logging.info(x))

    # Load data
    stock = 'MSFT'
    window_size = 3

    # QUI DOVRESTI PESCARE DAL TEST SET
    logging.info('Loading input_train, input_validation and input_test sets...')
    input_train = np.load(f'../data/input_train_{stock}_{window_size}_{N}days.npy', mmap_mode='r')
    input_val = np.load(f'../data/input_val_{stock}_{window_size}_{N}days.npy', mmap_mode='r')
    condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{N}days.npy', mmap_mode='r')
    condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{N}days.npy', mmap_mode='r')
    
    logging.info(f'input_train shape:\n\t{input_train.shape}')
    logging.info(f'condition_train shape:\n\t{condition_train.shape}')
    logging.info(f'input_val shape:\n\t{input_val.shape}')
    logging.info(f'condition_val shape:\n\t{condition_val.shape}')

    logging.info('Loading the scaler...')
    scaler = load(f'scaler_{N}days.joblib')
    logging.info('Done.')

    # Hyperparameters
    # T_condition = 2
    n_features = input_train.shape[2]
    T_gen = input_train.shape[1]
    latent_dim = 10

    gen_anomaly_scores = []
    disc_anomaly_score = []
    for window_train, window_condition in zip(input_train, condition_train):
        # add one dummy dimension to window_condition
        window_condition = np.expand_dims(window_condition, axis=0)
        # Find the latent space representation of the window
        res = explore_latent_space(window_train, window_condition, latent_dim, T_gen, generator)
        # Use it to generate a sample
        gen_output = generator(res)
        gen_output = tf.reshape(gen_output, [T_gen, n_features]).numpy()
        # Compute the difference wrt the original window
        diff = np.sum(np.abs(window_train - gen_output))
        # Compute the score given by the discriminator
        disc_output = discriminator(res)
        # Append the scores
        gen_anomaly_scores.append(diff)
        disc_anomaly_score.append(disc_output)
    
    gen_anomaly_scores = np.array(gen_anomaly_scores)
    disc_anomaly_score = np.array(disc_anomaly_score)

    # Plot the anomaly scores
    plt.figure()
    plt.plot(gen_anomaly_scores, label='Generator score')
    plt.plot(disc_anomaly_score, label='Discriminator score')
    plt.legend()
    plt.savefig(f'output_{job_id}_anomaly_scores.png')
        
        