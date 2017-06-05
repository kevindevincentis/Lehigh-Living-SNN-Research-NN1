import scipy.io as sio
from neuron import h, gui

vals = sio.loadmat('../MNIST/training_values.mat')
images = vals['images']

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(1)''')

img = [1]
h('numInputs = 1')
h.numInputs = len(img)
h('double img[numInputs]')

for i in range(len(img)):
    h.img[i] = img[i]

h('nn.input(&img)')

h('access nn.outCells[0].soma')

h.tstop = 15
print "About to RUN"

h.run()

try:
    input('Exit by pressing a key')
except: SyntaxError
