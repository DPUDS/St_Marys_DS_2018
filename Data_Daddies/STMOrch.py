import os
import pandas as pd 
import numpy as np 
from Logger import Log
from Tester import Test
from MLModeler import Modeler



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

for column in train.columns:
	print(column)
	train[column].replace(inplace=True,to_replace="None",value=0.0)

train.drop(["comment","category"], axis = 1, inplace =True)

Test(train,Modeler(specific_model = "RF"),MasterLog)

MasterLog.saveMasterLog()

