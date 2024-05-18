# -*- coding: utf-8 -*-
#Author: Solorzano, Juan Jose
#Date: July 18, 2023
#===============================================================================================================#
#Description: 
#===============================================================================================================#

import os
import math
import numpy as np
from misc import get_data,plot_data
from perceptron import Perceptron,FOLDER,training_loop,get_line_boundary,plot_loss

if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

if __name__ == '__main__':
    epochs,learning_rate,x_train,y_train,x_test,y_test,verbose = get_data()
    print('>> Using {} epochs'.format(epochs))
    print('>> Using {} learning rate'.format(learning_rate))
    #plot data before training.
    plot_data('train_data',x_train=x_train,y_train=y_train)
    plot_data('test_data',x_test=x_test)
    #Perceptron obj
    perceptron = Perceptron(num_inputs=2,learning_rate=learning_rate)
    cost_history = training_loop(perceptron,inputs=x_train,target=y_train,epochs=100)
    plot_loss(epochs=100,cost_history=cost_history)
    y_pred = []
    for x in x_test:
        y_pred.append(perceptron.predict(x))
    y_pred = np.array(y_pred)
    #plot predicted data.
    print('>> Accuracy: {}'.format(Perceptron.accuracy(y_test,y_pred)))
    plot_data('prediction',x_test,y_pred)
    _max = math.ceil(max(x_test.flatten()))
    _min = int(_max/2)
    get_line_boundary(perceptron,save=True,_min_=_min,_max_=_max)