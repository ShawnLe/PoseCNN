from keras.layers import Input, Dense
from keras.models import Model

input = Input(shape=[4])
probs = Dense(2, activation='sigmoid')(input)
probs_1 = Dense(2, activation='sigmoid')(probs)

model = Model(input=input, output=probs_1)
model.compile(optimizer='sgd', loss='binary_crossentropy')

# exit()

weights = model.trainable_weights # weight tensors
print("weights:")
for w in weights:
    print(w)

print("tt loss tensor {}".format(model.total_loss))

# weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable] # filter down weights tensors to only ones which are trainable
gradients = model.optimizer.get_gradients(model.total_loss, weights) # gradient tensors

print gradients
print weights
# ==> [dense_1_W, dense_1_b]


import keras.backend as K

input_tensors = [model.inputs[0], # input data
                 model.sample_weights[0], # how much to weight each sample by
                 model.targets[0], # labels
                 K.learning_phase(), # train or test mode
]

get_gradients = K.function(inputs=input_tensors, outputs=gradients)



from keras.utils.np_utils import to_categorical

inputs = [[[1, 2, 3, 4]], # X
          [1], # sample weights
          [[1,2]], # y
          0 # learning phase in TEST mode
]

# print zip(weights, get_gradients(inputs)[0])
print  get_gradients(inputs)[0] 
# ==> [(dense_1_W, array([[-0.42342907],
#                          [-0.84685814]], dtype=float32)),
#       (dense_1_b, array([-0.42342907], dtype=float32))]
