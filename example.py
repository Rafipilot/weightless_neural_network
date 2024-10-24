import Core

# Create a WNN with 3 neurons
wnn = Core.SimpleWNN(num_neurons=3)

# Train the network
training_data = [
    ([0, 0, 0, 0], 0),
    ([0, 0, 1, 1], 0),
    ([1, 1, 0, 0], 1),
    ([1, 1, 1, 1], 1)
]

for input_pattern, output in training_data:
    wnn.train(input_pattern, output)

# Test the network
test_input = [0, 0, 0, 1]
prediction = wnn.predict(test_input)

if prediction is not None:
    print(f"Prediction for {test_input} is: {prediction}")
else:
    print(f"No memory found for input {test_input}")