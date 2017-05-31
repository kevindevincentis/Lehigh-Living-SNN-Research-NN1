import scipy.io as sio
from neuron import h, gui

vals = sio.loadmat('../MNIST/training_values.mat')
images = vals['images']

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(28*28)''')

h('numInputs = 1')
h.numInputs = len(images[0])
h('double img[numInputs]')

for i in range(len(images[0])):
    h.img[i] = images[0][i]

h('nn.input(img)')

h('access nn.outCell.soma')

h.tstop = 30
print "About to RUN"

h.run()

try:
    input('Exit by pressing a key')
except: SyntaxError

h.run()
