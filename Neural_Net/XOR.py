import numpy as np

# Define the activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the XOR inputs and corresponding outputs
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Initialize random weights for the perceptron
input_size = 2
hidden_size = 2
output_size = 1

# Initialize weights with mean 0
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

# Set the learning rate
learning_rate = 0.1

# Train the perceptron
for epoch in range(10000):
    # Forward propagation
    input_layer = X
    hidden_layer_input = np.dot(input_layer, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)
    
    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)
    
    hidden_layer_error = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    
    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += input_layer.T.dot(d_hidden_layer) * learning_rate

# Test the trained perceptron
input_layer = X
hidden_layer_input = np.dot(input_layer, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_input)

# Round the output to 0 or 1
rounded_output = np.round(output)

# Calculate accuracy
accuracy = np.mean(rounded_output == y)

print("Output after training:")
print(output)
print("Rounded output:")
print(rounded_output)
print("Accuracy:", accuracy)
