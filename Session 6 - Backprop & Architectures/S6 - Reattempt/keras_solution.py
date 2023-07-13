import tensorflow as tf
from loguru import logger

from tensorflow.keras import datasets,layers, activations, losses, metrics, optimizers

logger.debug(f'{dir(datasets)}')
logger.debug(f'{dir(layers)}')
logger.debug(f'{dir(activations)}')
logger.debug(f'{dir(losses)}')
logger.debug(f'{dir(metrics)}')
logger.debug(f'{dir(optimizers)}')

train_data, test_data = datasets.mnist.load_data()
train_data_x, train_data_y = train_data
test_data_x, test_data_y = test_data
logger.debug(f'example: {train_data_x[0].shape}, entire train set {train_data_x.shape}')

logger.debug(f'{tf.keras.backend.image_data_format()}')
train_data_x = tf.expand_dims(train_data_x, axis = 3)
logger.debug(f'expand dims: {train_data_x.shape}')

model = tf.keras.models.Sequential(
    [
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(20, 7, activation="relu"),
        layers.Conv2D(30, 7, activation="relu"),
        layers.Conv2D(50, 7, activation="relu"),
        layers.Conv2D(80, 7, activation="relu"),
        layers.Flatten(),
        layers.Dense(10,activation="softmax")
    ]
)
logger.debug(f'{model(tf.random.normal(shape=(1,28,28,1)))}')
model.summary()

model.compile(loss=losses.MeanSquaredError(), optimizer= optimizers.SGD(learning_rate = 0.01) )
model.fit(train_data_x, train_data_y, batch_size=1, epochs = 3)

"""
1. Problem, Data, Model, Accuracy
2. Data Pattern Complexity, Model Complexity
"""