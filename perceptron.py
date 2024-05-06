# Author: Solorzano, Juan Jose
# Date:   
#
#
#

import numpy as np
import matplotlib.pyplot as plt
import os
"""
{{{
    a line is defined by:
        y=mx -> where m is the slope (gradient) and the x is the independient variable of the function.
        m=(p2/p1)*x
        a pint P' = (p'1,p'2) is on the line if the following condition is fulfiled:
        m*(p'1)-p'2 = 0
}}}
"""
FOLDER='images'
try:
    os.mkdir(FOLDER)
except:
    pass

def line():
#{{{
    plt.clf()
    # x vector
    x = list(range(10))
    p1 = 3.5 # coordinate a of a point.
    p2 = 1.0 # coordinate b of a point.
    plt.plot(p1,p2,'ro')
    px = 1.1# coordinate a of a point.
    py = 3.9# coordinate b of a point.
    plt.plot(px,py,'bo')
    # the point in the line
    P1_ = 4# coordinate a of a point.
    P2_ = 4.5 # coordinate a of a point.

    m = P2_/P1_ # slope
    #y = m*x line ecuation
    y = []
    for val in x: #fill the y vector with the slop.
        y.append(val*m)

    plt.plot(x,y) #plot the line
    plt.savefig('{}/line'.format(FOLDER))
#}}}

class Perceptron:
#{{{
    """
    @Public_class: This class is an example of the use of simple perceptron.
    @Attr:
        -
    """
    def __init__(self,learning_rate,epochs):
        self.weights = None
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _step_activation_function(self,z):
        """
        @<acces>_method: _description_
        @Args:
            - z (_type_): _description_
        @Returns (_type_): _description_ 
        """
        if isinstance(z,np.ndarray):
            array_values = np.array([1. if i>=0 else 0. for i in z])
            return array_values
        else: 
            if z >= 0:
                return 1
            else:
                return 0

    def fit(self,x,y):
        """
        @<acces>_method: _description_
        @Args:
            - x (_type_): _description_
            - y (_type_): _description_
        @Returns (_type_): _description_ 
        """
        n_features = x.shape[1] # num the inputs.
        self.weights = np.zeros((n_features)) # array_like: [0,0]
        # for loop for the training
        plt.clf()
        for epoch in range(self.epochs): #for each epoch defined.
            for i in range(len(x)):
                z = np.dot(x,self.weights) + self.bias #dot product plus bias value (row*column)
                y_pred = self._step_activation_function(z) # gets a predict value for each value in the array. 
                # weigth values actualization.
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i])*x[i]
                # bias value actulization.
                error = y[i]-y_pred[i] #error calculation for each value.
                self.bias = self.bias + self.learning_rate * error
            # plot the progress through each epoch.
            current_weight = self.weights[0]
            weight_to_plot = (current_weight*(-1))# change the position of the values in the graph.
            plt.plot(epoch,weight_to_plot,'bo') #plotting the points.
        plt.savefig('{}/learning_progress'.format(FOLDER)) # saving the image.
        return self.weights, self.bias # returns the final weights and bias calculated.

    def predict(self,x):
        z = np.dot(x,self.weights) + self.bias # 
        return self._step_activation_function(z)
#}}}

def dot_product(_A_,_B_):
    """
    @<acces>_method: _description_
    @Args:
        - _A_ (_type_): _description_
        - _B_ (_type_): _description_
    @Returns (_type_):
    Examples
    --------
    The dot product of two vectors in R_n is defined by:
        X*Y = x1y1 + x2y2 + ... + xnyn
    >>> X=[[2,7],[5,8]]
    >>> Y=[[1,5],[4,6]]
    >>> X1(2)*Y1(1) = 
    >>> X2(7)*Y2(5) = 35
    >>> X3(5)*Y3(4) = 20
    >>> X4(8)*Y4(6) = 48
    >>> ----------- + --
    >>>               103
    """
    _B_array = np.array(_B_)
    _A_array = np.array(_A_)