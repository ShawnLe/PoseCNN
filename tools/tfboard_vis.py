
import numpy as np
import matplotlib.pyplot as plt

# file = 'mdact_1.npy'

mdact0 = np.load('mdact_1.npy')
mdact1 = np.load('mdact1_1.npy')
inp = np.load('mdinp_12.npy')

for i in xrange(10):
    imgplot = plt.imshow(mdact0[0,:,:,i])
    plt.show()

for i in xrange(10):
    imgplot = plt.imshow(mdact1[:,:,i])
    plt.show()

print(inp.shape)
imgplot = plt.imshow(np.squeeze(inp).astype(np.int8))
plt.show()
