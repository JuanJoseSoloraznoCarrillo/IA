import pandas as pd
import numpy as np

#{{{
"""
iris_species_name = ['setosa',versicolor','virginica']
iris_species_num = [0,1,2] # 0=setosa; 1=versicolor; 2=virginica
iris_data_charateristics = ['sepal_length','sepal_width','petal_length',petal_width']
iris_data_values         = [[5.1,....,5.9],[3.5,...,3.0],[1.4,....,5.1],[0.2,...,1.8]]
A simple perceptron only can clasify between two values, then, we will symplify the iris dataset to two values.
"""
#}}}

# Data preparation using panda.
df = pd.read_csv('datasets/iris.csv') #read de .csv file.
df = df.sample(frac=1) # shuffle data information.
specie_names = df['species'].to_numpy()
setosa_num = np.array([1 if name=='setosa' else 0 for name in specie_names])
df = df.drop('species',axis=1) #remove the column four "species name"
numpy_df = df.to_numpy() # convert the pandas data frame in a numpy array.

x = numpy_df[:,(0,1)] #taking the columns 0 and 1. [all:(column_0,column_1)]
y = setosa_num

#train test split
train_percent = 0.5 #we'll use the 50% of all data.
total_data = len(x)
total_train_number = int(total_data*train_percent) # since x and y must be the same lenght, the total items taken for the training must be the same.
total_test_number = int(total_data-total_train_number)
x_train = x[:total_train_number]
y_train = y[:total_train_number]
x_test = x[total_test_number:total_data]
y_test = y[total_test_number:total_data]
