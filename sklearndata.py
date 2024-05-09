from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()
x = iris.data[:,(0,1)]
y = (iris.target == 0).astype(int)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=43)