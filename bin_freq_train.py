import scipy.io as sio
from matplotlib import pyplot
from back_prop import calc_weight_changes


def train_network(h):
    threshold = 0
    nIters = 1000
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

    # 2D matrix to store the weights
    weightVals = [ ([0] * 196) for neuron in range(10) ]
    for i in range(10):
        weightVals[i] = list(h.nn.outCells[i].getWeights())

    # Create hoc vectors for recording data
    t_vec = h.Vector()
    outputs = [0] * 10
    # for i in range(10):
    #     outputs.append(h.Vector())

    h("objref outputCounts[10]")
    outputs = [0] * 10
    # Loop through training images
    for cur in range(nIters):
        err = 1
        lastErr = 0
        while(abs(err - lastErr) >= .5):
            lastErr = err
            print "Training image: %d" %cur

            # input the image to the network
            for i in range(imgLen):
                h.img[i] = images[cur][i]

            h('nn.input(&img)')

            h.tstop = 80

            t_vec.record(h._ref_t)

            # begin recording the output neurons
            h('z = 0')
            for i in range(10):
                h.z = i
                # outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)
                h('nn.outCells[z].soma outputCounts[z] = new APCount(0.5)')
                h.outputCounts[i].thresh = threshold

            # Run simulation
            h.run()

            # Extract results from the output cells
            for i in range(len(outputs)):
                outputs[i] = h.outputCounts[i].n

            # Calculate the updates to the weights
            (updates, err) = train(outputs, labels[cur], int(h.nn.numNeurons), images[cur])

            # Update the weights of the network
            updateNeurons(h, updates, weightVals)


def train(outputs, truth, nWeights, img):
    # Compute the truth values based on "Perfect" scenarios
    truths = [0] * 10
    truths[truth] = 2

    return calc_weight_changes(outputs, truths, img)

# Update the weight values in the network
def updateNeurons(h, updates, weightVals):
    h('double update[nWeights]')
    for i in range(len(updates)):
        h.k = i

        for j in range(int(h.nn.numNeurons)):
            weightVals[i][j] += updates[i][j]
            h.update[j] = weightVals[i][j]
        h('nn.outCells[k].setWeights(&update)')
