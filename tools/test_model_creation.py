from keras.layers import Input, Dense, Conv1D, Conv2D
from keras.models import Model
from keras.layers import Lambda

import tensorflow as tf

# input = Input(shape=(4,), dtype='float32')
input = Input(shape=(480, 640, 3), dtype='float32', name='sl_input')

outputa = Conv2D(64, (3,3), name='sl_conv', padding='same')(input)
# output = Conv1D(1, 2, padding='same', name='sl_conv' )(input)
# output = input
# output = Lambda(lambda x: input, name='sl_output')

output = Conv2D(64, (3,3), name='sl_conv_a', padding='same')(outputa)

model = Model(input=input, output=output)
model.compile(optimizer='sgd', loss='binary_crossentropy')

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()
