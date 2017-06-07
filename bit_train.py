import scipy.io as sio
# from neuron import h, gui
from matplotlib import pyplot

def train_network(h):
    threshold = 20
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

    for cur in range(len(images)):
        print "Training image: %d" %cur

        for i in range(imgLen):
            h.img[i] = images[cur][i]

        h('nn.input(&img)')

        h.tstop = 15

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

        done = True
        foundWin = False
        t_truth = 0
        spike_times = [0] * len(outputs)
        for i in range(len(outputs[0])):
            done = True
            for j in range(10):
                if outputs[j][i] >= threshold:
                    if not foundWin:
                        foundWin = True
                        t_truth = i/40.0
                    if spike_times[j] == 0: spike_times[j] = i/40.0
                if spike_times[j] == 0: done = False
            if done: break

        updates = train(spike_times, labels[cur], int(h.nn.numNeurons))

        updateNeurons(h, updates)

def train(outputs, truth, nWeights):
    bestT = min(outputs)
    nNeurons = len(outputs)
    out = 0
    rate = 0.1
    weightDetlas = [ ([0] * nWeights) for neuron in range(nNeurons) ]
    posDelta = [rate] * nWeights
    negDelta = [-rate] * nWeights
    zeroDelta = [0] * nWeights

    for i in range(nNeurons):
        if (bestT == outputs[i]): out += 2**i

    truth = 1 << truth

    neg_change = (~truth) & out
    pos_change = truth & (~out)

    print(bin(neg_change), bin(pos_change))

    for i in range(nNeurons):
        mask = 1 << i
        if (mask & neg_change != 0): weightDetlas[i] = negDelta
        elif (mask & pos_change != 0): weightDetlas[i] = posDelta
        else: weightDetlas[i] = zeroDelta

    return weightDetlas

def updateNeurons(h, updates):
    h('double update[nWeights]')
    for i in range(len(updates)):
        h.k = i

        for j in range(int(h.nn.numNeurons)):
            h.update[j] = updates[i][j]
        h('nn.outCells[k].updateWeights(&update)')
