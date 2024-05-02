from perceptron import Perceptron,FOLDER
#from sklearndata import x_train,y_train,x_test,y_test
from irisdata import x_train,y_train,x_test,y_test
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

if not os.path.isdir(FOLDER):
    os.mkdir(FOLDER)

def _accuracy(y_true,y_pred,normalize=True):
    accuracy=[]
    for i in range(len(y_pred)):
        if y_pred[i]==y_true[i]:
            accuracy.append(1)
        else:
            accuracy.append(0)
    if normalize==True:
        return np.mean(accuracy)
    if normalize==False:
        return sum(accuracy)

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
    try:
        learning_rate = float(sys.argv[2])
    except:
        learning_rate = 0.001
    try:
        epochs = int(sys.argv[1])
    except:
        epochs = 100
    perceptron = Perceptron(learning_rate=0.001,epochs=epochs)
    perceptron.fit(x_train,y_train)
    y_pred = perceptron.predict(x_test)
    plot_data('prediction',x_test,y_pred)
