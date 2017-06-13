import scipy.io as sio
from matplotlib import pyplot
from back_prop import calc_weight_changes


def train_network(h):
    threshold = 0
    vals = sio.loadmat('../MNIST/training_values_compressed.mat')
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

    for cur in range(len(images)/6):
        for b in range(1):
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

            foundWin = False
            spike_freq = [0] * len(outputs)
            for i in range(len(outputs[0])):
                for j in range(10):
                    if outputs[j][i] >= threshold:
                        spike_freq[j] += 1.0/(h.tstop * 40)

            updates = train(spike_freq, labels[cur], int(h.nn.numNeurons), images[cur])

            updateNeurons(h, updates)

def train(outputs, truth, nWeights, img):
    truths = [.04] * 10
    truths[truth] = .31
    return calc_weight_changes(outputs, truths, img)

def updateNeurons(h, updates):
    h('double update[nWeights]')
    for i in range(len(updates)):
        h.k = i

        for j in range(int(h.nn.numNeurons)):
            h.update[j] = updates[i][j]
        h('nn.outCells[k].updateWeights(&update)')
