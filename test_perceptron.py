import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from perceptron import Perceptron,line,folder
import matplotlib.pyplot as plt
import os

if not os.path.isdir(folder):
    os.mkdir(folder)

def get_data_info():
    iris = load_iris()
    x = iris.data[:,(0,1)]
    y = (iris.target == 0).astype(np.int_)
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=42)
    for axis,group in zip(x_train,y_train):
        if group:
            color = 'ro'
        else:
            color = 'bo'
        plt.plot(axis[0],axis[1],color)
    plt.savefig('{}/training_data'.format(folder))
    plt.clf()
    for axis in x_test:
        plt.plot(axis[0],axis[1],'ro')
        plt.savefig('{}/test_data'.format(folder))

    return x_train,x_test,y_train,y_test

if __name__ == '__main__':
    x_train,x_test,y_train,y_test = get_data_info()
    perceptron = Perceptron(0.001,100)
    perceptron.fit(x_train,y_train)
    y_pred = perceptron.predict(x_test)
    plt.clf()
    for axis,group in zip(x_test,y_pred):
        if group:
            color = 'ro'
        else:
            color = 'bo'
        plt.plot(axis[0],axis[1],color)
        plt.savefig('{}/prediction'.format(folder))
    plt.clf()
    line()
