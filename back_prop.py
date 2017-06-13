# Code adpated from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
def calc_weight_changes(outputs, truths, inputs):
    LEARNING_RATE = 0.25
    nWeights = 196
    return train(outputs, truths, inputs, nWeights, LEARNING_RATE)

def train(outputs, truths, inputs, nWeights, LEARNING_RATE):
    # 1. Output neuron deltas
    nNeurons = len(outputs)
    pd_errors_wrt_output_neuron_total_net_input = [0] * nNeurons

    weightDetlas = [ ([0] * nWeights) for neuron in range(nNeurons) ]

    for o in range(nNeurons):
        pd_errors_wrt_output_neuron_total_net_input[o] = calculate_pd_error_wrt_total_net_input(truths[o], outputs[o])


    # 3. Update output neuron weights
    for o in range(nNeurons):
        for w_ho in range(nWeights):

            inp = inputs[w_ho]
            if (inp > 1.0/60): inp = 0.1*3200
            else: inp = 0.03*3200
            pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * inp

            weightDetlas[o][w_ho] -= LEARNING_RATE * pd_error_wrt_weight

    return weightDetlas

def calculate_total_error(training_sets):
    total_error = 0
    for t in range(len(training_sets)):
        training_inputs, truths = training_sets[t]
        for o in range(len(truths)):
            total_error += calculate_error(truths[o])
    return total_error

def calculate_pd_error_wrt_total_net_input(target_output, output):
    return calculate_pd_error_wrt_output(output, target_output) * calculate_pd_total_net_input_wrt_input(output);

# The error for each neuron is calculated by the Mean Square Error method:
def calculate_error(output, target_output):
    return 0.5 * (target_output - output) ** 2

def calculate_pd_error_wrt_output(output, target_output):
    return -(target_output - output)

def calculate_pd_total_net_input_wrt_input(output):
    return 1/3200.0
