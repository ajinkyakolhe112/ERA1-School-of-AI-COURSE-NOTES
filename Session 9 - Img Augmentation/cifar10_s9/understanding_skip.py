import tensorflow.keras as keras
from tensorflow.keras import datasets, models, layers, losses, metrics, optimizers

from loguru import logger
import wandb

# NICE CODE
(x_train, y_train), (x_val, y_val) = datasets.cifar10.load_data()

logger.debug(f'check: {x_train.shape}{y_train.shape}{x_val.shape}{y_val.shape}')
logger.debug(f'result: (50000, 32, 32, 3)(50000, 1)(10000, 32, 32, 3)')
logger.debug(f'TODO: reshape, y_train & y_val')

y_train_vector = keras.utils.to_categorical(y_train,10)
y_val_vector = keras.utils.to_categorical(y_val,10)
logger.debug(f'check: {y_train_vector[:5]}')
logger.debug(f'result: only 1 column is 1 in 5 vectors of 10 dims')

logger.debug(f'check: {x_train[0].T[0][0:2][0:2]}')
logger.debug(f'result: color range 0 - 255')

keras.models.Sequential([

])

