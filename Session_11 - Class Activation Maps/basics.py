import tensorflow as tf
from tensorflow import keras

"""
$$
y = \sigma(X,W)\\


$$
"""

model = keras.models.Sequential([
    keras.layers.Conv2D(32,100,"relu"),
    keras.layers.Conv2D(32,100,"relu"),
    keras.layers.Maxpool2D(2,2),
    keras.layers.Conv2D(10, 1, "relu"),

    keras.layers.Conv2D(64, 100, "relu"),
    keras.layers.Conv2D(64, 100, "relu"),
    keras.layers.Maxpool2D(2,2),
    keras.layers.Conv2D(10, 1, "relu"),

    keras.layers.Conv2D(256, 100, "relu"),
    keras.layers.Conv2D(128, 100, "relu"),
    keras.layers.Maxpool2D(2,2),
    keras.layers.Conv2D(10, 1, "relu"),

    keras.layers.Conv2D(1024, 100, "relu"),
    keras.layers.Conv2D(1024, 100, "relu"),
    keras.layers.Maxpool2D(2,2),
    keras.layers.Conv2D(10, 1, "relu"), 

    
    keras.layers.Conv2D(512, 1, "relu"),
    # Parallel: keras.layers.Dense(512, 512, "relu"), # equivalent to skip connection

    keras.layers.Conv2D(512, 16, "relu"), # final image to visualize 16 bits

    keras.layers.GlobalAveragePooling2D(512),


    keras.layers.Dense(512,512, "relu"), keras.layers.Conv2D(512, 1, "relu"),
    keras.layers.Dense(512,512, "relu"),
    keras.layers.Dense(512,10, "relu"),


])