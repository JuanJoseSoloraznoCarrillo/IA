# -*- coding: utf-8 -*-
#Author: Solorzano, Juan Jose
#Date: July 18, 2023
#Python: >3.6
#===============================================================================================================#
#Description: 
# The linear function that aggregates the inputs signals for a single neuron or precessing unit is defined as:
# Z = b+(sum(i=1,n){wi*xi}) ->  b+(w0*x0+w1*x1+...+wn*xn)
# The Z value output is used as input for the threshold function f(z). The b constant added at the beginning, the bias term,
# is a way to simplify learning a good threshold value for the network.kkkkkkkkkkkkkkkkkkkkkkkkkk
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
        X1--(W1)--|(perceptron)|                                      |
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
    
    def __init__(self,num_inputs,learning_rate=0.001):
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

    def _activation_function(self, input:list, _type_:str,threshold) -> np.ndarray:
        """
        Step function activation
        @Definition:
            >>> h(z) = {0 if z < Threshold; 1 if >= Threshold}
        """
        if _type_ in get_defined_activation_functions():
            function_name = _type_+'_function'
            if hasattr(__import__(__name__),function_name):
                activation_function = getattr(__import__(__name__),function_name)
                return activation_function(input,threshold)
            else:
                raise
        else:
            raise NameError(f'The "{_type_}" function is not defined yet !!!!!!')
    
    def predict(self,data:np.ndarray,activation_function:str,threshold=0) -> np.ndarray:
        """
        @<acces>_method: _description_
        @Args:
            - data (np.ndarray): _description_
        @Returns (np.ndarray): _description_ 
        @Definitions:
            >>> Z -> the weights sum.
        """
        if not isinstance(activation_function,str):
            raise NameError(f'The "{activation_function}" is not a string format!!!!!!!!!!!!!!!!')
        Z = np.dot(data,self.weights) + self.bias
        output = self._activation_function(input=Z,_type_=activation_function,threshold=threshold)
        return output

    @staticmethod
    def accuracy(y_true:list,y_pred:list) -> float:
        """Calculate accuracy"""
        correct = 0
        total = len(y_true)
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                correct += 1
        return (correct / total) * 100

def get_defined_activation_functions():
    return ['step','sigmoid','relu']



def step_function(Z:list,threshold=int):
    """
    @<acces>_method: _description_
    @Args:
        - Z (list): _description_
        - threshold (_type_): _description_
    @Returns
    """
    if Z >= threshold:
        return 1
    else:
        return 0

def relu_function(x:int,threshold:int):
    return np.maximum(threshold, x)

def sigmoid_function(x,threshold):
    return threshold / (threshold + np.exp(-x))

def training_loop(neural_obj:Perceptron,inputs:np.ndarray,target:np.ndarray,epochs:int) -> None:
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
    cost_history = [] #to plot the loss function.
    for _ in range(epochs): # for each epoch defined.
        errors = []
        for input,target in zip(inputs,targets):
            y_pred = neural_obj.predict(input)
            error = target - y_pred
            errors.append(error)
            weight_gradient = error*input
            bias_gradient = error
            neural_obj.weights += (neural_obj.learning_rate * weight_gradient)
            neural_obj.bias += neural_obj.learning_rate * bias_gradient
        np_errors = np.array(errors)
        cost = (np_errors ** 2).sum() / (2 * len(inputs))
        cost_history.append(cost)
    return cost_history

def get_line_boundary(perceptron:Perceptron,save=False,_min_=4,_max_=8)->None:
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
    m=((bias/-weights[1])/(bias/weights[0]))
    c=-bias/weights[1]
    x=range(_min_,_max_)
    y=m*x+c
    plt.plot(x, y, label='y = {}x + {}'.format(m, c),linestyle='--')
    if save:
        plt.savefig('{}/line_boundary'.format(FOLDER))

def plot_loss(epochs:int,cost_history:list) -> None:
    if not os.path.isdir(FOLDER):
        os.mkdir(FOLDER)
    plt.clf()
    plt.figure(figsize=(15, 10))
    plt.plot(range(1, epochs + 1), cost_history)
    plt.xlabel('Epoch',fontdict={'fontsize':'large'})
    plt.ylabel('Cost',fontdict={'fontsize':'large'})
    plt.title('Cost over Epochs',fontdict={'fontsize':'xx-large'})
    plt.grid(True)
    plt.savefig('{}/cost_function'.format(FOLDER))