# -*- coding: utf-8 -*-
#Author: Solorzano, Juan Jose
#Date: July 18, 2023
#===============================================================================================================#
#Description: 
#===============================================================================================================#

#from perceptron import FOLDER
FOLDER='images'
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_data(fig_name:str,x_train=None,y_train=None,x_test=None) -> None:
    if isinstance(x_test,np.ndarray):
        plt.clf()
        plt.figure(figsize=(15, 10))
        plt.xlabel('Sepal Lenght',fontdict={'fontsize':'large'})
        plt.ylabel('Sepal Width',fontdict={'fontsize':'large'})
        plt.title('Iris Setosa',fontdict={'fontsize':'xx-large'})
        plt.grid(True)
        for axis in x_test:
            plt.plot(axis[0],axis[1],'ro')
        plt.savefig('{}/{}'.format(FOLDER,fig_name))
    if isinstance(x_train,np.ndarray) and isinstance(y_train,np.ndarray):
        plt.clf()
        plt.figure(figsize=(15, 10))
        plt.xlabel('Sepal Lenght',fontdict={'fontsize':'large'})
        plt.ylabel('Sepal Width',fontdict={'fontsize':'large'})
        plt.title('Iris Setosa',fontdict={'fontsize':'xx-large'})
        plt.grid(True)
        for axis,group in zip(x_train,y_train):
            if group:
                color = 'ro'
            else:
                color = 'bo'
            plt.plot(axis[0],axis[1],color)
        plt.savefig('{}/{}'.format(FOLDER,fig_name))

def get_arg() -> argparse:
    console_arg = argparse.ArgumentParser()
    console_arg.add_argument('-e','--epochs',type=int,help='The epochs that will be used for the training phase.')
    console_arg.add_argument('-l','--lr',type=float,help='The learning rate used.')
    console_arg.add_argument('-v','--verbose',action='store_true',help='To show the information of the training process.')
    console_arg.add_argument('-d','--data',help='The data that will be used for the training.')
    args = console_arg.parse_args()
    return args

def get_data(data='irisdata'):
    args = get_arg()
    if args.data == 'sklearn' or data== 'sklearn':
        print('>> Using sklearn dataset')
        from sklearndata import x_train,y_train,x_test,y_test
    else:
        print('>> using iris dataset')
        from irisdata import x_train,y_train,x_test,y_test
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 250
    if args.lr:
        learning_rate = args.lr
    else:
        learning_rate = 0.001
    if args.verbose:
        verbose = True
    else:
        verbose = False
    return epochs,learning_rate,x_train,y_train,x_test,y_test,verbose