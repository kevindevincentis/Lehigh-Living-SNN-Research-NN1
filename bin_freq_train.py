import scipy.io as sio
# from neuron import h, gui
from matplotlib import pyplot

def train_network(h):
    threshold = -20
    vals = sio.loadmat('../MNIST/training_values.mat')
    images = vals['images']
    imgLen = len(images[0])
    labels = vals['labels']
    labels = labels[0]

    h('access nn.outCells[0].soma')

    h('k = 0')
    h('numInputs = 1')
    h.numInputs = imgLen
    h('double img[numInputs]')

    h('nWeights = nn.numNeurons')

    t_vec = h.Vector()
    outputs = list()
    for i in range(10):
        outputs.append(h.Vector())

    for cur in range(len(images)/600):
        print "Training image: %d" %cur

        for i in range(imgLen):
            h.img[i] = images[cur][i]

        h('nn.input(&img)')

        h.tstop = 80

        t_vec.record(h._ref_t)

        for i in range(10):
            outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)


        h.run()

        # Plot output
        # f, axarr = pyplot.subplots(10, sharex=True, sharey=True)
        # f.figsize = (16,16)
        # for i in range(len(axarr)):
        #     ax = axarr[i]
        #     ax.plot(t_vec, outputs[i])

        # done = True
        # foundWin = False
        # spike_freq = [0] * len(outputs)
        # for i in range(len(outputs[0])):
        #     done = True
        #     for j in range(10):
        #         if outputs[j][i] >= threshold:
        #             spike_freq[j] += 1
        #
        # print spike_freq

        updates = train(labels[cur], int(h.nn.numNeurons), images[cur])

        updateNeurons(h, updates)

def train(truth, nWeights, inputs):
    nNeurons = 10
    weights = [ ([0] * nWeights) for neuron in range(nNeurons) ]

    for j in range(nWeights):
        for i in range(nNeurons):
            if (inputs[j] > 1/60.0):
                if (i == truth): weights[i][j] = .0001
                else: weights[i][j] = -0.0001


    return weights

def updateNeurons(h, updates):
    h('double update[nWeights]')
    for i in range(len(updates)):
        h.k = i

        for j in range(int(h.nn.numNeurons)):
            h.update[j] = updates[i][j]
        h('nn.outCells[k].updateWeights(&update)')
