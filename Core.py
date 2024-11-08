class WeightlessNeuron:
    def __init__(self):
        self.memory_inputs = []
        self.memory_outputs = []

    def train(self, input, output):# Adding the Input output pairs to lists
        # Store the input and corresponding output in memory
        self.memory_inputs.append(input)
        self.memory_outputs.append(output)

    def hamming_distance(self, input1, input2): #Calculate hamming distance
        if len(input1) != len(input2):
            print("Error: input shapes don't match with stored memory")
            raise ValueError("Inputs must have the same length.")

        hamming_distance = 0
        for i in range(len(input1)): 
            if input1[i] != input2[i]:
                hamming_distance += 1 #Add 1 to the distance if we have a diffrence
        return hamming_distance

    def predict(self, input):
        distances = [] # Initialising list to temporary hold the distance between the Input and each stored Input
        
        for stored_input in self.memory_inputs:
            hamming_dist = self.hamming_distance(input, stored_input)
            distances.append(hamming_dist)
        
        # Find the closest match in memory
        closest_distance = min(distances)
        index = distances.index(closest_distance)
        
        # Return the corresponding output from memory_outputs
        output = self.memory_outputs[index]
        return 

neuron = WeightlessNeuron()

# Train the neuron with input-output pairs
neuron.train([1, 1], 1)
neuron.train([1, 0], 0)
print("Memory Inputs: ", neuron.memory_inputs)
print("Memory Outputs: ", neuron.memory_outputs)

# Predict the output for input [0, 0]
output = neuron.predict([0, 0])
print("Predicted Output:", output)
