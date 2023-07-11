import tensorflow as tf
import numpy as np

# Data or $X,Y$ , model or $f(,W)$, error_func = f(X_in,W) - Y_in
x = np.array([1,2,3,4,5,6],dtype=float)
y = np.array([100,150,200,250,300,350],dtype=float)

# IF: x = x.reshape(len(x),1) # x.ndim = 2. -> input_shape = [1,1]
# IF: x.ndim = 1 -> input_shape = [1] ( ndim 1 = vector = line any length)


model = tf.keras.models.Sequential(
	[
		# input shape of even a simple network can trip you up
		tf.keras.layers.Dense(units=1,input_shape=[1])
	])
try:
	model.weights
finally:
	pass
output = model(x)

error_func = tf.keras.losses.MeanSquaredError()
try:
	error_func(output,y)
finally:
	pass
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# at gradient of 0.001, changes are small enough to be around previous values
model.compile( loss= error_func, optimizer= optimizer)
model.fit(x,y, epochs=100)

def test_code():
	print(x.shape,y.shape)
	print(len(x),len(y))
	
	print(model(x))

	model.compile(loss="mean_squared_error",optimizer="sgd")
	model.fit(x,y,epochs=100)
	output = model.predict([7])
	print(output)

def reflection():
	"""
	advanced question. give upto 1000k x,y pair examples. 
		1. how many datapoints you need to get 95% accuracy, when training 1 epoch?
		2. how many datapoints you need to get 95% accuracy, when training 10 epochs?
	"""

	pass

if __name__=="__main__":
	test_code()