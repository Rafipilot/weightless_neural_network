class WN:  
    def __init__(self):
        self.memory_inputs = []
        self.memory_outputs = []

    def train(self, input, output):
        # Store the input and corresponding output in memory
        self.memory_inputs.append(input)
        self.memory_outputs.append(output)

    def predict(self, input):
        # Check if the input is in memory_inputs
        if input in self.memory_inputs:
            # Find the index of the input and return the associated output
            index = self.memory_inputs.index(input)
            return self.memory_outputs[index]
        else:
            # Return None if the input hasn't been encountered before
            return None


# Create an instance of the weightless neuron
neuron = WN()

# Train the neuron with input 1 and output 1
neuron.train(1, 1)
neuron.train(0, 0)
print("History: ", neuron.memory_inputs, neuron.memory_outputs)

# Predict the output for input 1
output = neuron.predict(0)
print("Output:", output)
