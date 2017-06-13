from bin_freq_train import train_network
import scipy.io as sio
from neuron import h, gui
from matplotlib import pyplot

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(14*14)''')

train_network(h)
allWeights = [ ([0] * 196) for neuron in range(10) ]
for i in range(10):
    allWeights[i] = list(h.nn.outCells[i].getWeights())

# pyplot.show()
allWeights = {'allWeights': allWeights}
sio.savemat('trained_weights', allWeights)
try:
    input('Exit by pressing a key')
except: SyntaxError
