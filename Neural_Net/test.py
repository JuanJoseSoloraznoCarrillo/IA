import numpy as np
import os
import matplotlib.pyplot as plt
import math

FOLDER = 'images'

class Perceptron:
    """
    @Public_class: This class is an example of the use of simple perceptron.
    @Attr:
              ____+(learning_rate*error)_______Weights_Update__________
              |                                                       |
              |   --------------                                      |
        X1--(W1)--|            |                                      |
         .        |            |                                      |
        X2--(W2)--| Sum(Xn*Wn) |--[activation]--->> Output --> (error)[y-y^]
         .        |            |
        Xn--(Wn)--|            |
                  --------------
                        |
            (bias)-------
        @Definition: 
            >>> Z = W1*X1+W1X2 + ... + WnXn -> X^T*W
    """
    
    def __init__(self, num_inputs, learning_rate=0.001):
        """
        @Constructor_method: Constructor for the Perceptron class
        @Args:
            - learning_rate (float): The learning rate used for the training phase.
            - epochs (int): Each epoch represents one pass through the entire training dataset.
            - verbose (bool): Flag to determine if some information will be shown during the train phase.
        @Params:
            - weights (np.ndarray): 
            - bias
        """
        self.inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()

    def _step_activation_function(self, Z, threshold=0):
        """
        Step function activation
        @Definition:
            >>> h(z) = {0 if z < Threshold; 1 if >= Threshold}
        """
        return np.array([1 if z >= threshold else 0 for z in Z])
    
    def _sigmoid_activation_function(self, Z, threshold=0):
        """
        Sigmoid activation function
        @Definition:
            >>> h(z) = 1 / (1 + exp(-z))
        """
        return 1 / (1 + np.exp(-Z))

    def predict(self, data, activation='step'):
        """
        @<acces>_method: _description_
        @Args:
            - data (np.ndarray): _description_
        @Returns (np.ndarray): _description_ 
        @Definitions:
            >>> Z -> the weights sum.
        """
        Z = np.dot(data, self.weights) + self.bias
        if activation == 'step':
            output = self._step_activation_function(Z)
        elif activation == 'sigmoid':
            output = self._sigmoid_activation_function(Z)
        return output

    @staticmethod
    def accuracy(y_true, y_pred):
        """Calculate accuracy"""
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return (correct / total) * 100

class SimpleNeuralNetwork:
    def __init__(self, n_neurons=2, n_inputs=2, learning_rate=0.001):
        """x
        @Constructor_method: _description_
        @Args:
            - n_neurons (int): _description_
            - n_inputs (int): _description_
        @Returns
        """
        #one hidden_layer
        self.learning_rate = learning_rate
        self.hidden_layer = [Perceptron(n_inputs) for _ in range(n_neurons)]
        self.output_layer = Perceptron(n_inputs)

    def predict(self, data,f):
        hidden_layer_prediction = [perceptron.predict(data, 'sigmoid') for perceptron in self.hidden_layer]
        out_data = np.array(hidden_layer_prediction)
        output_layer_prediction = self.output_layer.predict(out_data.T, 'sigmoid')
        return output_layer_prediction
        
def training_loop(neural_obj, inputs, target, epochs):
        inputs = np.array(inputs)
        target = np.array(target)
        cost_history = [] #to plot the loss function.
        for _ in range(epochs): # for each epoch defined.
            errors = []
            for idx in range(len(inputs)): 
                y_pred = neural_obj.predict(inputs[idx], f='sigmoid')
                error = target[idx] - y_pred
                errors.append(error)
                #Gradient descent
                weight_gradient = -error * inputs[idx]
                bias_gradient = -error
                # Weights and bias update.
                if isinstance(neural_obj, SimpleNeuralNetwork):
                    for perceptron in neural_obj.hidden_layer:
                        perceptron.weights -= perceptron.learning_rate * weight_gradient
                        perceptron.bias -= perceptron.learning_rate * bias_gradient
                    neural_obj.output_layer.weights -= neural_obj.learning_rate * weight_gradient
                    neural_obj.output_layer.bias -= neural_obj.learning_rate * bias_gradient
                else:
                    neural_obj.weights -= (neural_obj.learning_rate * weight_gradient)
                    neural_obj.bias -= neural_obj.learning_rate * bias_gradient
            np_errors = np.array(errors)
            cost = (np_errors ** 2).sum() / (2 * len(inputs))
            cost_history.append(cost)
        return cost_history

def get_line_boundary(perceptron, save=False, _min_=4, _max_=8):
    # (w · x) + b -> w:weights; x:inputs; b:bias
    # (w1*x1 + w2*x2) + b -> if x2=y
    # w1*x + w2*y + b -> ax+by+c=0
    #
    # x = -(w2*y/w1) - (b/w1) -> if y=0 => x=-b/w1 | x1 = -b/w1
    # y = -(w1*x/w2) - (b/w2) -> if x=0 => y=-b/w2 | x2 = -b/w2
    #
    # x=-b/w1 -> p1=(-b/w1,0) -> p1=x -> (-b/w1)m
    # y=-b/w2 -> p2=(0,-b/w2) -> p2=c -> c
    #
    # m = (0-(-b/w2))/(-b/w1 - 0) -> m=(b/w2)/(-b/w1)
    # y = ax+c | m=(b/w2)/(-b/w1) | c = -b/w2 
    # y = ((b/w2)/(-b/w1))x + (-b/w2)
    bias = perceptron.bias
    weights = perceptron.weights
    #line calculation.
    m = ((bias / -weights[1]) / (bias / weights[0]))
    c = -bias / weights[1]
    x = np.linspace(_min_, _max_, 100)
    y = m * x + c
    plt.plot(x, y, label='y = {}x + {}'.format(m, c), linestyle='--')
    if save:
        plt.savefig('{}/line_boundary'.format(FOLDER))

def plot_loss(epochs, cost_history):
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, epochs + 1), cost_history)
    plt.xlabel('Epoch', fontdict={'fontsize': 'large'})
    plt.ylabel('Cost', fontdict={'fontsize': 'large'})
    plt.title('Cost over Epochs', fontdict={'fontsize': 'xx-large'})
    plt.grid(True)
    plt.savefig('{}/cost_function'.format(FOLDER))

if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    neural_net = SimpleNeuralNetwork(n_neurons=2, n_inputs=2)
    cost_history = training_loop(neural_net, inputs=X, target=y, epochs=10000)
    plot_loss(10000, cost_history)
    y_pred = neural_net.predict(X,f='sigmoid')
    print('>> Accuracy: {}'.format(Perceptron.accuracy(y, y_pred)))
