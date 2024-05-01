import pandas as pd

df = pd.read_csv('datasets/iris.csv') #read de .csv file.
df = df.drop('species',axis=1) #remove the column four "species name"
numpy_df = df.to_numpy()
numpy_df[:,(0,1)] #taking the columns 0 and 1. [all:(column_0,column_1)]

