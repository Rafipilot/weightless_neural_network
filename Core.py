class WeightlessNeuron:
    def __init__(self):
        self.memory_inputs = []
        self.memory_outputs = []

    def train(self, input, output):
        # Store the input and corresponding output in memory
        self.memory_inputs.append(input)
        self.memory_outputs.append(output)

    def hamming_distance(self, input1, input2):
        if len(input1) != len(input2):
            print("Error: input shapes don't match with stored memory")
            raise ValueError("Inputs must have the same length.")

        # Correctly calculate Hamming distance
        hamming_distance = 0
        for i in range(len(input1)):  # Compare corresponding elements
            if input1[i] != input2[i]:
                hamming_distance += 1
        return hamming_distance

    def predict(self, input):
        # Check if the input is in memory_inputs
        distances = []
        
        for stored_input in self.memory_inputs:
            hamming_dist = self.hamming_distance(input, stored_input)
            distances.append(hamming_dist)
        
        # Find the closest match in memory
        closest_distance = min(distances)
        index = distances.index(closest_distance)
        
        # Return the corresponding output from memory_outputs
        output = self.memory_outputs[index]
        return output

# Create an instance of the weightless neuron
neuron = WeightlessNeuron()

# Train the neuron with input-output pairs
neuron.train([1, 1], 1)
neuron.train([1, 0], 0)
print("Memory Inputs: ", neuron.memory_inputs)
print("Memory Outputs: ", neuron.memory_outputs)

# Predict the output for input [0, 0]
output = neuron.predict([0, 0])
print("Predicted Output:", output)
