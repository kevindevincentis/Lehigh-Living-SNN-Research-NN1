import scipy.io as sio
from neuron import h, gui

vals = sio.loadmat('../MNIST/training_values_compressed.mat')
images = vals['images']
labels = vals['labels']
labels = labels[0]

weights = sio.loadmat('trained_weights.mat')
weights = weights['allWeights']


h('''load_file("network.hoc")
objref nn
nn = new fullLayer(14*14)''')

h('nWeights = nn.numNeurons')
h('double update[nWeights]')
h('k = 0')

# for i in range(len(weights)):
#     h.k = i
#
#     for j in range(int(h.nn.numNeurons)):
#         h.update[j] = weights[i][j]
#     h('nn.outCells[k].setWeights(&update)')

cur = 0

img = images[cur]
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
    outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)

h.run()

foundWin = False
threshold = -20
spike_freq = [0] * len(outputs)
for i in range(len(outputs[0])):
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

h("print nn.outCells[0].nclist.object(10).weight")
try:
    input('Exit by pressing a key')
except: SyntaxError
