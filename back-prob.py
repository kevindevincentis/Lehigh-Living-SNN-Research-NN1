# Code adpated from https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

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

def calculate_total_error(self, training_sets):
    total_error = 0
    for t in range(len(training_sets)):
        training_inputs, truths = training_sets[t]
        self.feed_forward(training_inputs)
        for o in range(len(truths)):
            total_error += self.output_layer.neurons[o].calculate_error(truths[o])
    return total_error

def calculate_output(self, inputs):
    self.inputs = inputs
    self.output = self.squash(self.calculate_total_net_input())
    return self.output

def calculate_total_net_input(self):
    total = 0
    for i in range(len(self.inputs)):
        total += self.inputs[i] * self.weights[i]
    return total + self.bias

# Apply the logistic function to squash the output of the neuron
# The result is sometimes referred to as 'net' [2] or 'net' [1]
def squash(self, total_net_input):
    return 1 / (1 + math.exp(-total_net_input))

# Determine how much the neuron's total input has to change to move closer to the expected output
#
# Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
# the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
# the partial derivative of the error with respect to the total net input.
# This value is also known as the delta (δ) [1]
# δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
#
def calculate_pd_error_wrt_total_net_input(target_output, output):
    return calculate_pd_error_wrt_output(output, target_output) * calculate_pd_total_net_input_wrt_input(output);

# The error for each neuron is calculated by the Mean Square Error method:
def calculate_error(output, target_output):
    return 0.5 * (target_output - output) ** 2

# The partial derivate of the error with respect to actual output then is calculated by:
# = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
# = -(target output - actual output)
#
# The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
# = actual output - target output
#
# Alternative, you can use (target - output), but then need to add it during backpropagation [3]
#
# Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
# = ∂E/∂yⱼ = -(tⱼ - yⱼ)
def calculate_pd_error_wrt_output(output, target_output):
    return -(target_output - output)

# The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
# yⱼ = φ = 1 / (1 + e^(-zⱼ))
# Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
#
# The derivative (not partial derivative since there is only one variable) of the output then is:
# dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
def calculate_pd_total_net_input_wrt_input(output):
    return output * (1 - output)

# The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
# = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
#
# The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
# = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
def calculate_pd_total_net_input_wrt_weight(self, index):
    return self.inputs[index]

def calc_weight_changes(outputs, truth):
