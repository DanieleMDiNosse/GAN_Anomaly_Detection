import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
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
    parser.add_argument('-nlg', '--n_layers_gen', help='Number of generator layers', type=int)
    parser.add_argument('-nld', '--n_layers_disc', help='Number of discriminator layers', type=int)
    parser.add_argument('-tg', '--type_gen', help='Type of generator model (conv, lstm, dense)', type=str)
    parser.add_argument('-td', '--type_disc', help='Type of discriminator model (conv, lstm, dense)', type=str)
    parser.add_argument('-c', '--condition', action='store_true', help='Conditioning on the first T_condition time steps')
    parser.add_argument('-tc', '--T_condition', help='Number of time steps to condition on', type=int, default=2)

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

    # Initialize logger
    logging.basicConfig(filename=f'output_{job_id}_anomalydet.log', format='%(message)s', level=levels[args.log])
    
    args = parser.parse_args()
    paths = os.listdir('models')
    path = [p for p in paths if args.job_id in p]

    if len(path) == 0:
        raise ValueError('Job id not found')
    elif len(path) > 1:
        raise ValueError('Multiple job ids found')

    path = path[0]
    model_path = [p for p in os.listdir(f'models/{path}') if 'h5' in p]
    gen_model_path = [p for p in os.listdir(f'models/{path}') if 'generator_model' in p]
    disc_model_path = [p for p in os.listdir(f'models/{path}') if 'discriminator_model' in p]
    # Load the models
    generator = tf.keras.models.load_model(f'models/{path}/{gen_model_path[0]}')
    discriminator = tf.keras.models.load_model(f'models/{path}/{disc_model_path[0]}')

    # Load data
    stock = 'MSFT'
    window_size = 3
    day = 0

    logging.info('Loading input_train, input_validation and input_test sets...')
    input_train = np.load(f'../data/input_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
    input_val = np.load(f'../data/input_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
    condition_train = np.load(f'../data/condition_train_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
    condition_val = np.load(f'../data/condition_val_{stock}_{window_size}_{day+1}.npy', mmap_mode='r')
    
    logging.info(f'input_train shape:\n\t{input_train.shape}')
    logging.info(f'condition_train shape:\n\t{condition_train.shape}')
    logging.info(f'input_val shape:\n\t{input_val.shape}')
    logging.info(f'condition_val shape:\n\t{condition_val.shape}')
    
    logging.info('Loading the scaler...')
    scaler = load(f'scaler_{day+1}_{job_id}_{args.type_gen}_{args.type_disc}_{args.n_layers_gen}_{args.n_layers_disc}_{args.data}_{args.T_condition}_{args.loss}.joblib')
    logging.info('Done.')

    # Hyperparameters
    T_condition = args.T_condition
    n_features = input_train.shape[2]
    T_gen = input_train.shape[1] - T_condition
    latent_dim = 10

    diffs = []
    anomaly_scores = []
    for window in input_train:
        res = explore_latent_space(window, latent_dim, T_condition, T_gen, generator)
        gen_output = generator(res)
        disc_output = discriminator(res)

        gen_output = tf.reshape(gen_output, [T_gen, n_features]).numpy()
        diff = np.abs(window[T_condition:] - gen_output)
        diffs.append(diff)
        