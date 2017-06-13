import scipy.io as sio
from neuron import h, gui

vals = sio.loadmat('../MNIST/training_values.mat')
images = vals['images']

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(28*28)''')

img = [1.0/20] * 784
h('numInputs = 1')
h.numInputs = len(img)
h('double img[numInputs]')

print(images[0][0], images[0][1], images[0][2], images[0][3])

for i in range(len(img)):
    h.img[i] = img[i]

h('nn.input(&img)')

h('access nn.outCells[0].soma')

h.tstop = 80
print "About to RUN"

t_vec = h.Vector()
t_vec.record(h._ref_t)

outputs = list()
for i in range(10):
    outputs.append(h.Vector())

for i in range(10):
    outputs[i].record(h.nn.cellArray[i].soma(0.5)._ref_v)

h.run()

foundWin = False
threshold = -20
spike_freq = [0] * len(outputs)
for i in range(len(outputs[0])):
    for j in range(10):
        if outputs[j][i] >= threshold:
            spike_freq[j] += 1

print spike_freq

try:
    input('Exit by pressing a key')
except: SyntaxError
