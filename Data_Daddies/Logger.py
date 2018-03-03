import datetime as dt 
import pandas as pd 
import os


# NEEDS TO BE UPDATED FOR FRAUD DATA

class Log():

	# Collection Log tracks the information about the datasets 
	# Being collected by API and ensures that all meta-data information 
	# is stored.
	def __init__(self, master_log_name, 
					   res_log_name, 
					   masterColNames, 
					   resColNames):

		#Log names
		self.MasterLogName = master_log_name
		self.ResLogName = res_log_name

		#Two DataFrame logs for performance and data collection
		self.MasterLog = pd.DataFrame(columns = masterColNames)
		self.ResLog = pd.DataFrame(columns = resColNames)

	# Add MasterLog record
	def addMasterLogRecord(self, test):
		new_record_df = pd.DataFrame(
								      [[
								       #Test Execution Information
								       test.ExecutionDateStart,
								       test.ExecutionTimeStart,
								       test.ModelDuration,
								       test.TestDuration,

								       #Information about training and testing data
								       test.Modeler.TestRatio,
								       test.Modeler.TrainRowNum,
								       test.Modeler.TestRowNum,

								       #Information from modeler
								       test.Modeler.Classifiers,
								       test.Modeler.SVMParams,
								       test.Modeler.RFEstimators,
								       test.Modeler.KNNeighbors,
								       test.Modeler.MonteCarlo,
								       test.Modeler.MonteCarloSampSize,

								       #Modeler information about Support vector machine
								       test.Modeler.SVMPerf[0],		#Accuracy
								       test.Modeler.SVMPerf[1],		#SSE


								       #Modeler information about Random forest
								       test.Modeler.RFPerf[0],		#Accuracy
								       test.Modeler.RFPerf[1],		#SSE


								       #Modeler information about Gaussian Naive Bayes
								       test.Modeler.GNBPerf[0],		#Accuracy
								       test.Modeler.GNBPerf[1],		#SSE

								       
								       #Modeler information about K-Nearest Neighbors
								       test.Modeler.KNNPerf[0],		#Accuracy
								       test.Modeler.KNNPerf[1],		#SSE


								       #Modeler information about Logistic Regression
								       test.Modeler.LOGPerf[0],		#Accuracy
								       test.Modeler.LOGPerf[1],		#SSE

								       #Results Log filename for the modeler
								       test.Modeler.ResLogFilename]],

									   #Add the Collection Log Column Names
									   columns = self.MasterLog.columns)

		self.MasterLog = pd.concat([self.MasterLog ,new_record_df], axis = 0)
		self.MasterLog.reset_index(drop = True, inplace = True)

	def addResultRecord(self, model):
		new_metadata_df = pd.DataFrame(
								      [[
								      	#Test Execution Information
								      	dt.datetime.now().date(),
								       	dt.datetime.now().time(),
								       	model.ModelDurationSec,

								       	#Train Data Information
								       	model.TestRatio,
										model.TrainRowNum,			#Total Rows train
										model.TestRowNum,			#Total Rows test

										#Model General Information
										model.ModelName,

										#Model performance information
								       	model.ModelPerf[0],		#Accuracy
								       	model.ModelPerf[1]		#Sum of sqaured errors
								        ]],

									   #Add the Collection Log Column Names
									   columns = self.ResLog.columns)

		self.ResLog = pd.concat([self.ResLog ,new_metadata_df], axis = 0)
		self.ResLog.reset_index(drop = True, inplace = True)


	# Save the collection log as a csv
	def saveMasterLog(self):
		#Change working directory for Master Logs
		os.chdir("/Users/Sam/Documents/Python/St_Marys_DS_2018/Data/MasterLogs")

		self.MasterLog.to_csv(str(dt.datetime.now().strftime("%m-%d_%H.%M.%S")) + "-MasterLog.csv", sep = ",")

	# Save the results log as a csv
	def saveResultsLog(self, resLogName):
		#Change working directory for Result Logs
		os.chdir("/Users/Sam/Documents/Python/St_Marys_DS_2018/Data/ResLogs")

		#Save the log
		self.ResLog.to_csv(resLogName, sep = ",")



