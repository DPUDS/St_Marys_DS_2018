import os
import pandas as pd 


os.chdir("/Users/Sam/Desktop")

train = pd.read_csv("STM_train.csv", sep = "	")

print(train.head())

print(train.category)