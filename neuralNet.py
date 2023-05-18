# Devon Griffith
# Jan 12, 2017
# First attempt at machine learning
# with a neural network
# This NN has 3 layers, first with ?? neurons, second with ?? neurons, then with a single neuron.
# Supervised Multi-layered Feed-Forward Perceptron for Classification

from numpy import array, random, dot, abs, exp, ones
from numpy.random import binomial

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1
        self.number_of_neurons = number_of_neurons

class NeuralNetwork():

    def __init__(self, layer_1, layer_2, layer_3):
        random.seed(1)
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.layer_3 = layer_3


    def __activation(self, x):
        return 1 / (1 + exp(-x))

    def __activation_prime(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, number_of_iterations):
        for iteration in range(number_of_iterations):

            output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.think(training_inputs, do_dropout=False)

            layer_3_error = training_outputs - output_from_layer_3
            layer_3_delta = layer_3_error * self.__activation_prime(output_from_layer_3)

            layer_2_error = layer_3_delta.dot(self.layer_3.synaptic_weights.T)
            layer_2_delta = layer_2_error * self.__activation_prime(output_from_layer_2)

            layer_1_error = layer_2_delta.dot(self.layer_2.synaptic_weights.T)
            layer_1_delta = layer_1_error * self.__activation_prime(output_from_layer_1)

            layer_1_adjustment = training_inputs.T.dot(layer_1_delta)
            layer_2_adjustment = output_from_layer_1.T.dot(layer_2_delta)
            layer_3_adjustment = output_from_layer_2.T.dot(layer_3_delta)

            self.layer_1.synaptic_weights += layer_1_adjustment
            self.layer_2.synaptic_weights += layer_2_adjustment
            self.layer_3.synaptic_weights += layer_3_adjustment


    def think(self, inputs, do_dropout=False):
        # if(do_dropout):
        #     dropout_percent = 0.2
        # else:
        #     dropout_percent = 0.0

        output_from_layer_1 = self.__activation(dot(inputs, self.layer_1.synaptic_weights))
        # if(do_dropout):
        #     output_from_layer_1 *= binomial([ones(len(inputs), NeuronLayer.number_of_neurons)], 1-dropout_percent)[0]*(1.0/(1-dropout_percent))
        output_from_layer_2 = self.__activation(dot(output_from_layer_1, self.layer_2.synaptic_weights))
        output_from_layer_3 = self.__activation(dot(output_from_layer_2, self.layer_3.synaptic_weights))

        return output_from_layer_1, output_from_layer_2, output_from_layer_3

    def print_weights(self):
        print("    Layer 1 (4 neuron(s), with 3 input(s)): ")
        print(self.layer_1.synaptic_weights)
        print("    Layer 2 (3 neuron(s), with 4 input(s)):")
        print(self.layer_2.synaptic_weights)
        print("    Layer 3 (1 neuron(s), with 3 input(s)): ")
        print(self.layer_3.synaptic_weights)


if __name__ == "__main__":

    random.seed(5)

    layer_1 = NeuronLayer(4, 3) # (4 neuron(s), with 3 input(s))
    layer_2 = NeuronLayer(3, 4) # (3 neuron(s), with 4 input(s))
    layer_3 = NeuronLayer(1, 3) # (1 neuron(s), with 3 input(s))

    alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    neural_network = NeuralNetwork(layer_1, layer_2, layer_3)  # Initialize the neural network

    print(" Without training...")
    hidden_state, hidden_state_2, test = neural_network.think(array([1, 1, 1]), )
    print("Considering situation [1, 1, 1 -> ?:]")
    print("Target: ")
    print("[1]")
    print("Guess: ")
    print(test)

    print("Beginning training")

    print("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # 8 examples, each consisting of 3 input values and 1 output value
    training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 0, 0], \
                                 [0, 0, 0], [1, 1, 1]])
    training_set_outputs = array([[0, 1, 0, 0, 1, 1, 1, 1]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 100000)  # Train neural network 100,000 times

    print("\nStage 2) New synaptic weights after training: \n")
    neural_network.print_weights()

    # Test the neural network with a new situation.
    print("\nStage 3) Considering a new situation: [1, 0, 0] -> ?: ")
    hidden_state, hidden_state_2, final_answer = neural_network.think(array([1, 0, 0]))
    print("Target:\n[1]\nGuess:")  # Target: [0]
    print(final_answer)
