import scipy.io as sio
from neuron import h, gui

vals = sio.loadmat('../MNIST/training_values.mat')
images = vals['images']

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(28*28)''')

img = images[0]
img[1] = 1.0/20
h('numInputs = 1')
h.numInputs = len(img)
h('double img[numInputs]')

print(images[0][0], images[0][1], images[0][2], images[0][3])

for i in range(len(img)):
    h.img[i] = img[i]

h('nn.input(&img)')

h('access nn.outCells[0].soma')

h.tstop = 85
print "About to RUN"

h.run()

try:
    input('Exit by pressing a key')
except: SyntaxError
