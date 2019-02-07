# Convolutional Layers in Keras
# To create a convolutional layer in Keras, you must first import the necessary module:
from keras.layers import Conv2D

# Then, you can create a convolutional layer by using the following format:
Conv2D(filters, kernel_size, strides, padding, activation='relu', input_shape)

# Arguments
# You must pass the following arguments:

# filters - The number of filters.
# kernel_size - Number specifying both the height and width of the (square) 
#   convolution window.

# There are some additional, optional arguments that you might like to tune:

# strides - The stride of the convolution. If you don't specify anything, strides
#   is set to 1.
# padding - One of 'valid' or 'same'. If you don't specify anything, padding is
#   set to 'valid'.
# activation - Typically 'relu'. If you don't specify anything, no activation is
#   applied. You are strongly encouraged to add a ReLU activation function to
#   every convolutional layer in your networks.
# NOTE: It is possible to represent both kernel_size and strides as either a
#   number or a tuple.

# When using your convolutional layer as the first layer (appearing after the 
# nput layer) in a model, you must provide an additional input_shape argument:

# input_shape - Tuple specifying the height, width, and depth (in that order) of
#   the input.
# NOTE: Do not include the input_shape argument if the convolutional layer is not
#   the first layer in your network.

# There are many other tunable arguments that you can set to change the behavior
# of your convolutional layers. To read more about these, we recommend perusing
# the official documentation.

Conv2D(filters=16, kernel_size=2, strides=2, activation='relu', input_shape=(200, 200, 1))
Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
Conv2D(64, (2,2), activation='relu')

# Max Pooling Layers in Keras
# To create a max pooling layer in Keras, you must first import the necessary module:

from keras.layers import MaxPooling2D

#Then, you can create a convolutional layer by using the following format:
MaxPooling2D(pool_size, strides, padding)

# Arguments
# You must include the following argument:

# pool_size - Number specifying the height and width of the pooling window.

# There are some additional, optional arguments that you might like to tune:

# strides - The vertical and horizontal stride. If you don't specify anything,
#   strides will default to pool_size.
# padding - One of 'valid' or 'same'. If you don't specify anything, padding is
#   set to 'valid'.
# OTE: It is possible to represent both pool_size and strides as either a number
#   or a tuple.

# You are also encouraged to read the official documentation.

# Example
# Say I'm constructing a CNN, and I'd like to reduce the dimensionality of a
#   convolutional layer by following it with a max pooling layer. Say the
#   convolutional layer has size (100, 100, 15), and I'd like the max pooling
#   layer to have size (50, 50, 15). I can do this by using a 2x2 window in my
#   max pooling layer, with a stride of 2, which could be constructed in the
#   following line of code:

MaxPooling2D(pool_size=2, strides=2)

# If you'd instead like to use a stride of 1, but still keep the size of the window at 2x2, then you'd use:

MaxPooling2D(pool_size=2, strides=1)

# Checking the Dimensionality of Max Pooling Layers

from keras.models import Sequential
from keras.layers import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
model.summary()
