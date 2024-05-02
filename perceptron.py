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
    @Usage:
    @theroy:
        learning rate: 
    """
    def __init__(self,learning_rate,epochs):
        self.weights = 0
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self,z):
        return np.heaviside(z,0)
    
    def fit(self,x,y):
        n_features = x.shape[1]
        self.weights = np.zeros((n_features))
        
        for epoch in range(self.epochs):
            for i in range(len(x)):
                z = np.dot(x,self.weights) + self.bias
                y_pred = self.activation(z)
                self.weights = self.weights + self.learning_rate * (y[i] - y_pred[i])*x[i]
                self.bias = self.bias + self.learning_rate * (y[i] - y_pred[i])

        return self.weights, self.bias

    def predict(self,x):
        z = np.dot(x,self.weights) + self.bias
        return self.activation(z)
#}}}

