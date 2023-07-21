import tensorflow as tf

from tensorflow.keras import datasets, preprocessing, models, layers, losses, metrics, optimizers

import wandb
from loguru import logger

train_data, test_data = datasets.mnist.load_data()

model = models.Sequential()
model.add()
model.compile(
    loss = "",
    optimizer= "",
    metrics = ["",]
)

model.fit(x_train,y_train)
# easy network. But can't monitor batch by batch. Hence pytorch