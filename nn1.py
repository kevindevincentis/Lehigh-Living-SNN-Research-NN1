import scipy.io as sio
from neuron import h, gui
from matplotlib import pyplot

threshold = 0
vals = sio.loadmat('../MNIST/training_values.mat')
images = vals['images']

h('''load_file("network.hoc")
objref nn
nn = new fullLayer(28*28)''')

h('numInputs = 1')
h.numInputs = len(images[0])
h('double img[numInputs]')

for i in range(len(images[0])):
    h.img[i] = images[0][i]

h('nn.input(&img)')

h('access nn.outCells[0].soma')

h.tstop = 15
print "About to RUN"

t_vec = h.Vector()             # Time stamp vector
t_vec.record(h._ref_t)


outputs = list()
for i in range(10):
    outputs.append(h.Vector())
    outputs[i].record(h.nn.outCells[i].soma(0.5)._ref_v)


h.run()

f, axarr = pyplot.subplots(10, sharex=True, sharey=True)
f.figsize = (16,16)
for i in range(len(axarr)):
    ax = axarr[i]
    ax.plot(t_vec, outputs[i])
    # ax.xlabel('time (ms)')
    # ax.ylabel('outCells[0] mV')

done = True
foundWin = False
t_truth = 0
spike_times = [0] * len(outputs)
for i in range(len(outputs[0])):
    done = True
    for j in range(10):
        if outputs[j][i] >= threshold:
            if not foundWin:
                print "Winner is digit %d at %dms" %(j, i/40.0)
                foundWin = True
                t_truth = i/40.0
            if spike_times[j] == 0: spike_times[j] = i/40.0
        if spike_times[j] == 0: done = False
    if done: break

truths = [0] * len(outputs)
for i in range(len(outputs)):
    if i == labels[0]: truths[i] = t_truth
    else: truths[i] = t_truth + 5


# pyplot.show()

try:
    input('Exit by pressing a key')
except: SyntaxError
