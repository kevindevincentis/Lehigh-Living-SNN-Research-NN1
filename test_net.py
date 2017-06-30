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
weights = sio.loadmat('./trained_weights')
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
h("objref outputCounts[10]")

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

    outputs = [0] * 10

    threshold = 0
    h('z = 0')
    for i in range(10):
        h.z = i
        # outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)
        h('nn.outCells[z].soma outputCounts[z] = new APCount(0.5)')
        h.outputCounts[i].thresh = threshold

    h('access nn.outCells[0].soma')
    # Run simulation
    h.run()

    # Extract results from the output cells

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

    if winners[0] == truth:
        wins += 1.0
    print "Wins rate: %d" %(wins)
    print "Trials: %d" %(cur)

    print "Off by: %d" %(outputs[winners[0]] - outputs[truth])

    confusion[winners[0]][truth] += 1

print('\n'.join([''.join(['{:4}'.format(item) for item in row])
      for row in confusion]))

try:
    input('Exit by pressing a key')
except: SyntaxError
