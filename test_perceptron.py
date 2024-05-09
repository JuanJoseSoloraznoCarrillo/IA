# -*- coding: utf-8 -*-
#Author: Solorzano, Juan Jose
#Date: July 18, 2023
#===============================================================================================================#
#Description: 
#===============================================================================================================#

from perceptron import Perceptron,FOLDER
from misc import get_data,plot_data
import os
import matplotlib.pyplot as plt

if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

if __name__ == '__main__':
    epochs,learning_rate,x_train,y_train,x_test,y_test,verbose = get_data()
    print('>> Using {} epochs'.format(epochs))
    print('>> usnig {} learning rate'.format(learning_rate))
    #plot data before training.
    plot_data('train_data',x_train=x_train,y_train=y_train)
    plot_data('test_data',x_test=x_test)
    #Perceptron obj
    perceptron = Perceptron(learning_rate=learning_rate,epochs=epochs,verbose=verbose)
    perceptron.train(x_train,y_train)
    y_pred = perceptron.predict(x_test)
    #plot predicted data.
    print('>> Accuracy: {}'.format(Perceptron.accuracy(y_test,y_pred)))
    plot_data('prediction',x_test,y_pred)
    
    perceptron.get_line_bundary()
    plt.savefig('{}/line'.format(FOLDER))