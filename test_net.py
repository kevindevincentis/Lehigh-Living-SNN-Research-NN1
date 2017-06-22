import scipy.io as sio
from neuron import h, gui
from matplotlib import pyplot

# Create Network
h('''load_file("network.hoc")
objref nn
nn = new fullLayer(14*14)''')

# Load in the testing images and labels
vals = sio.loadmat('../MNIST/training_values_compressed.mat')
images = vals['images']
imgLen = len(images[0])
labels = vals['labels']
labels = labels[0]

# Load in trained weights
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

wins = 0.0
# Set up a confusion matrix to keep track of performance
confusion = [ ([0] * 10) for neuron in range(10) ]
nTrials = 100
# Begin looping through the testing images
for cur in range(nTrials):
    # Give the input image to the network
    for i in range(imgLen):
        h.img[i] = images[cur][i]

    h('nn.input(&img)')

    h.tstop = 100

    # Set up recording vectors to look at outputs
    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    outputs = list()
    for i in range(10):
        outputs.append(h.Vector())

    for i in range(10):
        outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)

    h('access nn.outCells[0].soma')
    # Run simulation
    h.run()

    # Extract results from the output cells
    threshold = 0
    spike_freq = [0] * len(outputs)
    for i in range(len(outputs[0])):
        done = True
        for j in range(10):
            if outputs[j][i] >= threshold:
                spike_freq[j] += 1

    print spike_freq

    best_freq = max(spike_freq)

    # Update metrics
    winners = list()
    for i in range(len(spike_freq)):
        if (spike_freq[i] == best_freq): winners.append(i)

    truth = labels[cur]

    print "TRUTH: %d " %truth
    print "WINNERS: ",
    print winners

    if winners[0] == truth:
        wins += 1.0
    print "Wins rate: %d" %(wins)
    print "Trials: %d" %(cur)

    print "Off by: %d" %(spike_freq[winners[0]] - spike_freq[truth])

    confusion[winners[0]][truth] += 1

print('\n'.join([''.join(['{:4}'.format(item) for item in row])
      for row in confusion]))

try:
    input('Exit by pressing a key')
except: SyntaxError
