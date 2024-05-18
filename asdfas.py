import numpy as np
from perceptron import Perceptron

class SimpleNeuralNet:
    def __init__(self, n_neurons=2, n_inputs=2, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.hidden_layer = [Perceptron(n_inputs) for _ in range(n_neurons)]
        self.output_layer = Perceptron(n_neurons)

    def predict(self, data):
        hidden_layer_output = [relu(np.dot(data, perceptron.weights) + perceptron.bias) for perceptron in self.hidden_layer]
        output_data = np.array(hidden_layer_output)
        output_layer_prediction = self.output_layer.predict(output_data, activation_function='sigmoid', threshold=0.5)
        return output_layer_prediction

def training_loop(neural_obj: SimpleNeuralNet, inputs: np.ndarray, target: np.ndarray, epochs: int) -> list:
    inputs = np.array(inputs)
    targets = np.array(target)
    learning_rate = neural_obj.learning_rate
    cost_history = []
    for _ in range(epochs):
        errors = []
        for input_data, target_data in zip(inputs, targets):
            y_pred = neural_obj.predict(input_data)
            error = target_data - y_pred
            errors.append(error)
            weight_gradient = error * input_data
            bias_gradient = error
            for perceptron in neural_obj.hidden_layer:
                perceptron.weights += learning_rate * weight_gradient
                perceptron.bias += learning_rate * bias_gradient
            neural_obj.output_layer.weights += learning_rate * weight_gradient
            neural_obj.output_layer.bias += learning_rate * bias_gradient
        np_errors = np.array(errors)
        cost = (np_errors ** 2).sum() / (2 * len(inputs))
        cost_history.append(cost)
    return cost_history

def calculate_accuracy(y_true, y_pred):
    return np.mean(np.round(y_pred) == y_true)

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    neural_net = SimpleNeuralNet(n_neurons=4, n_inputs=2, learning_rate=0.1)  # Increase the number of neurons in the hidden layer
    cost_history = training_loop(neural_net, inputs=X, target=y, epochs=10000)
    y_pred = np.array([1 if neural_net.predict(x) > 0.5 else 0 for x in X])
    accuracy = calculate_accuracy(y, y_pred)
    print('>> Accuracy: {:.2f}%'.format(accuracy * 100))
    print("y_pred =", y_pred)
