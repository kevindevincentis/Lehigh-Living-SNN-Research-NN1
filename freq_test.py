import scipy.io as sio
from neuron import h, gui

vals = sio.loadmat('../MNIST/training_values_compressed.mat')
images = vals['images']
labels = vals['labels']
labels = labels[0]

weights = sio.loadmat('trained_weights.mat')
weights = weights['allWeights']

print weights[0]
h('''load_file("network.hoc")
objref nn
nn = new fullLayer(14*14)''')

h('nWeights = nn.numNeurons')
h('double update[nWeights]')
h('k = 0')

# weights = [([0.0008] * 196) for neuron in range(10)]
for i in range(len(weights)):
    h.k = i

    for j in range(int(h.nn.numNeurons)):
        h.update[j] = weights[i][j]
    h('nn.outCells[k].setWeights(&update)')

cur = 0

img = images[0]
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

outputs = [0] * 10
# for i in range(10):
#     outputs.append(h.Vector())
#

h("objref outputCounts[10]")

threshold = 0
h('z = 0')
for i in range(10):
    h.z = i
    # outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)
    h('nn.outCells[z].soma outputCounts[z] = new APCount(0.5)')
    h.outputCounts[i].thresh = threshold


h.run()

threshold = 0

for i in range(len(outputs)):
    outputs[i] = h.outputCounts[i].n

print outputs

best_freq = max(outputs)
winners = list()
for i in range(len(outputs)):
    if (outputs[i] == best_freq): winners.append(i)

truth = labels[cur]

print "TRUTH: %d " %truth
print "WINNERS: ",
print winners

try:
    input('Exit by pressing a key')
except: SyntaxError
