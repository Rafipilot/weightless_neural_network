class WN:
    def __init__(self):
        self.memory = {}

    def train(self, input, output):
        self.memory[tuple(input)] = output

    def predict(self, input):
        return self.memory.get(tuple(input), None)
    

class SimpleWNN:
    def __init__(self, num_neurons):
        # Initialize multiple RAM neurons
        self.neurons = [WN() for x in range(num_neurons)]

    def train(self, input_pattern, output):
        # Split the input pattern and train each neuron on part of the pattern
        for i, neuron in enumerate(self.neurons):
            neuron.train(input_pattern[i::len(self.neurons)], output)

    def predict(self, input_pattern):
        # Collect predictions from each neuron
        votes = []
        for i, neuron in enumerate(self.neurons):
            prediction = neuron.predict(input_pattern[i::len(self.neurons)])
            if prediction is not None:
                votes.append(prediction)
        
        # Majority vote to determine final output
        if votes:
            return max(set(votes), key=votes.count)  # Return most common output
        return None


