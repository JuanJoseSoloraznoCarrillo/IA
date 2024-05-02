from perceptron import Perceptron,FOLDER
#from sklearndata import x_train,y_train,x_test,y_test
from cldset import x_train,y_train,x_test,y_test
import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

def plot_data(fig_name,x_train=None,y_train=None,x_test=None):
    if isinstance(x_test,np.ndarray):
        plt.clf()
        for axis in x_test:
            plt.plot(axis[0],axis[1],'ro')
        plt.savefig('{}/{}'.format(FOLDER,fig_name))
    if isinstance(x_train,np.ndarray) and isinstance(y_train,np.ndarray):
        plt.clf()
        for axis,group in zip(x_train,y_train):
            if group:
                color = 'ro'
            else:
                color = 'bo'
            plt.plot(axis[0],axis[1],color)
        plt.savefig('{}/{}'.format(FOLDER,fig_name))

if __name__ == '__main__':
    print('>> [*] start data preparation')
    plot_data('train_data',x_train=x_train,y_train=y_train)
    plot_data('test_data',x_test=x_test)
    perceptron = Perceptron(0.001,100)
    perceptron.fit(x_train,y_train)
    y_pred = perceptron.predict(x_test)
    plot_data('prediction',x_test,y_pred)
