import scipy.io as sio
from neuron import h, gui
from matplotlib import pyplot

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(28*28)''')

vals = sio.loadmat('../MNIST/testing_values.mat')
images = vals['images']
imgLen = len(images[0])
labels = vals['labels']
labels = labels[0]

weights = sio.loadmat('./trained_weights.mat')
weights = weights['allWeights']
print weights
h('nWeights = nn.numNeurons')
h('double update[nWeights]')
h('k = 0')

for i in range(len(weights)):
    h.k = i

    for j in range(int(h.nn.numNeurons)):
        h.update[j] = weights[i][j]
    h('nn.outCells[k].setWeights(&update)')

h('numInputs = 1')
h.numInputs = imgLen
h('double img[numInputs]')

for cur in range(10):
    for i in range(imgLen):
        h.img[i] = images[cur][i]

    h('nn.input(&img)')

    h.tstop = 15

    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    outputs = list()
    for i in range(10):
        outputs.append(h.Vector())

    for i in range(10):
        outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)

    h.run()

    done = True
    foundWin = False
    threshold = 20
    spike_freq = [0] * len(outputs)
    for i in range(len(outputs[0])):
        done = True
        for j in range(10):
            if outputs[j][i] >= threshold:
                spike_freq[j] += 1

    print spike_freq

    best_freq = max(spike_freq)

    winners = list()
    for i in range(len(spike_freq)):
        if (spike_freq[i] == best_freq): winners.append(i)

    truth = labels[cur]

    print "TRUTH: %d " %truth
    print "WINNERS: ",
    print winners
try:
    input('Exit by pressing a key')
except: SyntaxError
