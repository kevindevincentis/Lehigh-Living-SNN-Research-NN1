# Code adpated from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
def calc_weight_changes(outputs, truths, inputs):
    LEARNING_RATE = 0.5
    nWeights = len(inputs[0])
    return train(output, truths, inputs, nWeights, LEARNING_RATE)

def train(outputs, truths, inputs, nWeights, LEARNING_RATE):
    # 1. Output neuron deltas
    nNeurons = len(outputs)
    pd_errors_wrt_output_neuron_total_net_input = [0] nNeurons

    weightDetlas = [ ([0] * nWeights) for neuron in range(nNeurons) ]
    for o in range(nNeurons):

        # ∂E/∂zⱼ
        pd_errors_wrt_output_neuron_total_net_input[o] = calculate_pd_error_wrt_total_net_input(truths[o], outputs[o])


    # 3. Update output neuron weights
    for o in range(nNeurons):
        for w_ho in range(nWeights):

            # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
            pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * inputs[o][w_ho]

            # Δw = α * ∂Eⱼ/∂wᵢ
            weightDetlas -= LEARNING_RATE * pd_error_wrt_weight

def calculate_total_error(training_sets):
    total_error = 0
    for t in range(len(training_sets)):
        training_inputs, truths = training_sets[t]
        for o in range(len(truths)):
            total_error += calculate_error(truths[o])
    return total_error

# Apply the logistic function to squash the output of the neuron
# The result is sometimes referred to as 'net' [2] or 'net' [1]

def calculate_pd_error_wrt_total_net_input(target_output, output):
    return calculate_pd_error_wrt_output(output, target_output) * calculate_pd_total_net_input_wrt_input(output);

# The error for each neuron is calculated by the Mean Square Error method:
def calculate_error(output, target_output):
    return 0.5 * (target_output - output) ** 2

def calculate_pd_error_wrt_output(output, target_output):
    return -(target_output - output)

def calculate_pd_total_net_input_wrt_input(output):
    return output * (1 - output)
