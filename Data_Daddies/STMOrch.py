import os
import pandas as pd 
import numpy as np 
from Logger import Log
import time
from Tester import Test
from MLModeler import Modeler
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA



os.chdir("/Users/Sam/Desktop")

train = pd.read_csv("STM_train.csv", sep = "	")


#############################################################################################################################

# Master Log for entire experimental study metadata. 
MasterLog = Log("Test_Master_Log", "Test_Results_Log", 

									#Masterlog column names
								  [[
								  	#Execution Metadata
								    "Execution_Date",
									"Execution_Time",
									"Modeler_Duration_Sec",
									"Execution_Duration_Sec",
									
									#Train and test information
									"Test_Ratio",
									"Train_Row_Num",
									"Test_Row_Num",

									#Modeler Meta-parameters
									"Models_Used",
									"SVM_Params",
									"RF_Estimators",
									"KNN_Neighbors",
									"Monte_Carlo_Bool",
									"Monte_Carlo_Samp_Size",

									#Modeler SVM Performance
									"SVM_Accuracy", 
									"SVM_SSE",
									
									#Model Random Forest Performance   
									"RF_Accuracy", 
									"RF_SSE", 
								

									#Model Gaussian Naive Bayes performance
									"GNB_Accuracy", 
									"GNB_SSE",
									

									#Model KNN performance
									"KNN_Accuracy", 
									"KNN_SSE",
									

									#Model Logistic performance 
									"LOG_Accuracy", 
									"LOG_SSE",

									#Results log filename
									"Res_Log_Filename"]],
			
									#Results column names
									["Results_Log"])

#############################################################################################################################


#print(train.head())

comment = train.comment
train.drop(["comment","id","category"], axis = 1, inplace =True)





#print(train.category)
to_dummy_columns = [] #['age_cat','category' 'stay_cat','sex', 'stay_cat', 'lang', 'er', 'category']
for column in to_dummy_columns:
	#print("\n\n\n" + column)
	train = pd.concat([
       train.iloc[:, : train.columns.get_loc(column)],
       pd.get_dummies(train[column], prefix=column),
       train.iloc[:, train.columns.get_loc(column) + 1 :]], 
       axis=1)

train.er = pd.to_numeric(train.er , errors = 'coerce')
train.er.fillna(2, inplace = True)

#Clean data temporarily, in a bad way
for column in train.columns:
	train[column] = pd.to_numeric(train[column], errors = 'coerce')
	train[column].fillna(train[column].median(), inplace = True)

#print(train.columns)

sent = pd.read_csv("/Users/Sam/Documents/Python/St_Marys_DS_2018/Data_Daddies/Data/sent.csv")
#print(sent.columns)
score = train.score
neg = (sent.Negative / (sent.Negative + sent.Positive))
pos = (sent.Positive / (sent.Negative + sent.Positive))
neg.fillna(0, inplace = True)
pos.fillna(0, inplace = True)
#train["negative"] = neg
#train["positive"] = pos
train.drop(["score"], axis = 1, inplace = True)
train["score"] = score

#Drop category
#train.drop("category", axis = 1, inplace = True)
#print(train.columns)



# train_sc_1 = train[train.stay_cat == 1]
# train_sc_2 = train[train.stay_cat == 2]
# train_sc_3 = train[train.stay_cat == 3]
# train_sc_4 = train[train.stay_cat == 4]
# train_sc_5 = train[train.stay_cat == 5]
# #train_sc_med = trainData[train.stay_cat == train.stay_cat.median()]

# train_list = [train_sc_1,
# 			  train_sc_2,
# 			  train_sc_3,
# 			  train_sc_4,
# 			  train_sc_5]

# #Random test
# for train in train_list:
# 	Test(train,Modeler(specific_model = "RF",n_estimators = 10,n_neighbors = 50, test_ratio = 0.1),MasterLog)

#print(train[train.score <= 5].describe())

# trainOld = train[train.age_cat >= 4]
# trainYoung = train[train.age_cat < 4]

# for i in range(11):
# 	print("\n\n\n\n" + str(i) + "\n\n\n\n", train[train.score == i].describe())


#############################################################################################################

cols = ['category_admn', 'category_disch', 'category_issues', 'category_meals',
       'category_nurses', 'category_overall', 'category_physn',
       'category_room', 'category_tests', 'category_visit']

# for name in cols:
# 	Test(train[train[name] == 1],Modeler(specific_model = "LOG",n_estimators = 30,n_neighbors = 50, test_ratio = 0.1),MasterLog)
print(train.dtypes)



#############################################################################################################


Test(train,Modeler(specific_model = "RF",n_estimators = 30,n_neighbors = 50, test_ratio = 0.1),MasterLog)
#Test(trainYoung,Modeler(specific_model = "RF",n_estimators = 10,n_neighbors = 50, test_ratio = 0.1),MasterLog)


MasterLog.saveMasterLog()

