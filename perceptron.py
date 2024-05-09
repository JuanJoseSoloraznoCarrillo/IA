# -*- coding: utf-8 -*-
#Author: Solorzano, Juan Jose
#Date: July 18, 2023
#Python: 3.6>
#===============================================================================================================#
#Description: This script demonstrates the head of a Python file with additional metadata.
# A line is defined by:
#    y=mx -> where m is the slope (gradient) and the x is the independient variable of the function.
#    m=(p2/p1)*x
#    a point P' = (p'1,p'2) is on the line if the following condition is fulfiled:
#    m*(p'1)-p'2 = 0
#===============================================================================================================#
import numpy as np
import matplotlib.pyplot as plt
import os
FOLDER='images'

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
    
    def __init__(self, learning_rate:float, epochs:int, verbose=False):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose

    def _step_activation_function(self, Z:list, threshold=0) -> np.ndarray:
        """
        Step function activation
        @Definition:
            >>> h(z) = {0 if z < Threshold; 1 if >= Threshold}
        """
        return np.array([1 if z>=threshold else 0 for z in Z])

    def train(self, inputs:np.ndarray, target:np.ndarray) -> None:
        inputs = np.array(inputs)
        target = np.array(target)
        n_features = inputs.shape[1] # num the inputs of the perceptron.
        self.weights = np.zeros(n_features) # array_like: [0,0]
        self.bias = 0
        self.cost_history = [] #to plot the loss function.
        for epoch in range(self.epochs): # for each epoch defined.
            errors = []
            for idx in range(len(inputs)): 
                y_pred = self.predict(inputs)
                error = target[idx] - y_pred[idx]
                errors.append(error)
                #Gradient descent
                weight_gradient = -error * inputs[idx]
                bias_gradient = -error
                self.weights -= (self.learning_rate * weight_gradient)
                self.bias -= self.learning_rate * bias_gradient
            np_errors = np.array(errors)
            cost = (np_errors ** 2).sum() / (2 * len(inputs))
            self.cost_history.append(cost)
            self.get_line_bundary()
            if self.verbose:
                print(f"Epoch {epoch + 1}: Weights = {self.weights}")
        #plot results
        plt.savefig('{}/line'.format(FOLDER))
        self.plot_loss()

    def predict(self, data:np.ndarray) -> np.ndarray:
        Z = np.dot(data, self.weights) + self.bias
        return self._step_activation_function(Z)

    def get_line_bundary(self):
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
        m=((self.bias/-self.weights[1])/(self.bias/self.weights[0]))
        c=-self.bias/self.weights[1]
        x=range(4,8)
        y=m*x+c
        plt.plot(x, y, label='y = {}x + {}'.format(m, c),linestyle='--')

    def plot_loss(self) -> None:
        if not os.path.isdir(FOLDER):
            os.mkdir(FOLDER)
        plt.clf()
        plt.plot(range(1, self.epochs + 1), self.cost_history)
        plt.xlabel('Epoch')
        plt.ylabel('Cost')
        plt.title('Cost over Epochs')
        plt.grid(True)
        plt.savefig('{}/tst_cost_fnc'.format(FOLDER))
        plt.show()

    @staticmethod
    def accuracy(y_true:list, y_pred:list) -> float:
        """Calculate accuracy"""
        correct = 0
        total = len(y_true)
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct += 1
        return (correct / total) * 100
