class WN:  #weightless neuron class
    def __init__(self):
        self.memory = {}

    def train(self, input, output):
        self.memory[tuple(input)] = output

    def predict(self, input):
        return self.memory.get(tuple(input), None)
    
class RAMNeuronLayer:
    def __init__(self, num_neurons, input_size_per_neuron):
        # Initialize the neurons in the layer
        self.neurons = [WN() for _ in range(num_neurons)]
        self.input_size_per_neuron = input_size_per_neuron
    
    def train(self, input_pattern, output):
        # Split the input pattern for each neuron in the layer
        for i, neuron in enumerate(self.neurons):
            sub_pattern = input_pattern[i * self.input_size_per_neuron:(i + 1) * self.input_size_per_neuron]
            neuron.train(sub_pattern, output)
    
    def predict(self, input_pattern):
        # Collect predictions from each neuron
        layer_output = []
        for i, neuron in enumerate(self.neurons):
            sub_pattern = input_pattern[i * self.input_size_per_neuron:(i + 1) * self.input_size_per_neuron]
            prediction = neuron.predict(sub_pattern)
            if prediction is not None:
                layer_output.append(prediction)
            else:
                layer_output.append(0) 
        return layer_output


class WNN():
    def __init__(self, input_size, num_neurons_layer1, num_neurons_layer2):
        # First layer with neurons that process the input
        self.layer1 = RAMNeuronLayer(num_neurons_layer1, input_size // num_neurons_layer1)
        # Second layer that processes the output of the first layer
        self.layer2 = RAMNeuronLayer(num_neurons_layer2, num_neurons_layer1)

    def train(self, input, output):
        #train Layer 1 with the raw input
        self.layer1.train(input, output)
        
        # predict the outputs from Layer 1 (since Layer 2 needs these as inputs)
        layer1_output = self.layer1.predict(input)
        
        # train Layer 2 with Layer 1's output
        self.layer2.train(layer1_output, output)

    def predict(self, input):
        layer1_output = self.layer1.predict(input)
        final_output = self.layer2.predict(layer1_output)
        return max(set(final_output), key=final_output.count)  # Majority 