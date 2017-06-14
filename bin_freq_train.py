import scipy.io as sio
from matplotlib import pyplot
from back_prop import calc_weight_changes


def train_network(h):
    threshold = -20
    nIters = 10000
    # Load training images and labels
    vals = sio.loadmat('../MNIST/training_values_compressed.mat')
    images = vals['images']
    imgLen = 196
    labels = vals['labels']
    labels = labels[0]

    h('access nn.outCells[0].soma')

    h('k = 0')
    h('numInputs = 1')
    h.numInputs = imgLen
    h('double img[numInputs]')

    h('nWeights = nn.numNeurons')

    # Create hoc vectors for recording data
    t_vec = h.Vector()
    outputs = list()
    for i in range(10):
        outputs.append(h.Vector())

    # Loop through training images
    for cur in range(nIters):
        for b in range(1):
            print "Training image: %d" %cur

            # input the image to the network
            for i in range(imgLen):
                h.img[i] = images[cur][i]

            h('nn.input(&img)')

            h.tstop = 80

            t_vec.record(h._ref_t)

            # begin recording the output neurons
            for i in range(10):
                outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)

            # Run simulation
            h.run()

            # Extract results from the output cells
            spike_freq = [0] * len(outputs)
            for i in range(len(outputs[0])):
                for j in range(10):
                    if outputs[j][i] >= threshold:
                        spike_freq[j] += 1.0/(h.tstop * 40)

            # Calculate the updates to the weights
            updates = train(spike_freq, labels[cur], int(h.nn.numNeurons), images[cur])

            # Update the weights of the network
            updateNeurons(h, updates)


def train(outputs, truth, nWeights, img):
    # Compute the truth values based on "Perfect" scenarios
    truths = [0] * 10
    truths[truth] = .31

    return calc_weight_changes(outputs, truths, img)

# Update the weight values in the network
def updateNeurons(h, updates):
    h('double update[nWeights]')
    for i in range(len(updates)):
        h.k = i

        for j in range(int(h.nn.numNeurons)):
            h.update[j] = updates[i][j]
        h('nn.outCells[k].updateWeights(&update)')
