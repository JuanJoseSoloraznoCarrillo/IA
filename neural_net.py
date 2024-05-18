import numpy as np
from perceptron import Perceptron

class SimpleNeuralNet:

    def __init__(self,n_neurons=2,n_inputs=2,learning_rate=0.001) -> None:
        self.learning_rate = learning_rate
        self.hidden_layer = [Perceptron(n_inputs) for _ in range(n_neurons)]
        self.output_layer = Perceptron(n_inputs)
    
    def predict(self,data):
        hidden_layer_prediction = [perceptron.predict(data,activation_function='relu',threshold=1) for perceptron in self.hidden_layer]
        output_data = np.array(hidden_layer_prediction)
        output_layer_prediction = self.output_layer.predict(output_data,activation_function='sigmoid',threshold=1)
        return output_layer_prediction

def training_loop(neural_obj:SimpleNeuralNet,inputs:np.ndarray,target:np.ndarray,epochs:int) -> None:
    """
    @<acces>_method: _description_
    @Args:
        - neural_obj (Perceptron): _description_
        - inputs (np.ndarray): _description_
        - target (np.ndarray): _description_
        - epochs (int): _description_
    @Returns (_type_): _description_ 
    """
    inputs = np.array(inputs)
    targets = np.array(target)
    learning_rate = neural_obj.learning_rate
    cost_history = [] #to plot the loss function.
    for _ in range(epochs): # for each epoch defined.
        errors = []
        for input,target in zip(inputs,targets):
            y_pred = neural_obj.predict(input)
            error = target - y_pred
            errors.append(error)
            weight_gradient = error*input
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

if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    neural_net = SimpleNeuralNet(n_neurons=2,n_inputs=2,learning_rate=0.001)
    cost_history = training_loop(neural_net, inputs=X, target=y, epochs=100)
    y_pred = []
    for x in X:
        y_pred.append(neural_net.predict(x))
    y_pred = np.array(y_pred)
    print(y_pred)
    print('>> Accuracy: {}'.format(Perceptron.accuracy(y, y_pred)))