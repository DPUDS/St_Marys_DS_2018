import pandas as pd 
import numpy as np 
import datetime as dt
from Logger import Log
import copy
from sklearn.model_selection import train_test_split

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Support Vector Machine model
from sklearn.svm import SVC

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Stochastic Gradient Descent Optimizer
from sklearn.linear_model import SGDClassifier

# Gaussian Naive Bayes Optimizer
from sklearn.naive_bayes import GaussianNB

# Machine learning logger for performance 
MLLog = Log("Master_Log",  "Results_Log",  
								#Masterlog column names
							  	["Master_Log"],

								#Results column names
								[[
								#Execution Metadata
								"Execution_Date",
								"Execution_Time",
								"Execution_Duration_Sec",

								#Sample metadata
								"Test_Ratio",
								"Train_Row_Num",
								"Test_Row_Num",

								#Model metadata
								"Model_Name",

								#Model performance
								"Accuracy", 
								"Sum_Squared_Error"]])

#################################################################################################################################

# Class takes a dataset sample and runs all ML analysis on it, storing the results
class Modeler():

	def __init__(self,
						sample = None,
					    test_ratio = 0.4,		
						n_neighbors = 5, 					
						SVMparams = ('rbf',1,5), 			
						n_estimators = 30,					
						monte_carlo = True,
						monte_carlo_samp_size = 5,
						specific_model = None):

		#ML Model parameters
		self.KNNeighbors = n_neighbors
		self.SVMParams = SVMparams
		self.RFEstimators = n_estimators

		# Performance and logistic information
		self.TestRatio = test_ratio
		self.MonteCarlo = monte_carlo
		self.MonteCarloSampSize = monte_carlo_samp_size
		
		#Logging information
		self.Log = copy.deepcopy(MLLog)
		self.ResLogFilename = str(dt.datetime.now().strftime("%m_%d")) + "-" + str(dt.datetime.now().strftime("%H.%M.%S")) + "-ResLog.csv"

		#Specific model information
		self.SpecificModel = specific_model

	def setSample(self,sample):
		self.Sample = sample

	def run_model(self):

		#General Bank of Classifiers (used with  "All")
		classifiers = [self.SVM_train_test_model,
			           self.RF_train_test_model,
			           self.GNB_train_test_model,
			           self.KNN_train_test_model,
			           self.LOG_train_test_model]

		#Dynamically Change what classifiers are used
		if self.SpecificModel is None:
			self.Classifiers = "All"
		else:
			self.Classifiers = self.SpecificModel

		#Update result log filename
		self.ResLogFilename = self.Classifiers + "-" + self.ResLogFilename

		# IF monte carlo analysis is asked for
		if self.MonteCarlo:
			self.model_engine(classifiers)

	'''
	This is the sklearn KNN model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def KNN_train_test_model(self, X_train, X_test, y_train, y_test):
		KNN_clf = KNeighborsClassifier(n_neighbors = self.KNNeighbors)
		KNN_clf.fit(X_train,y_train)
		predicted = KNN_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	This is the sklearn SVM model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def SVM_train_test_model(self, X_train, X_test, y_train, y_test):
		SVM_clf = SVC(kernel = self.SVMParams[0], C = self.SVMParams[1], gamma = self.SVMParams[2])
		SVM_clf.fit(X_train,y_train)
		predicted = SVM_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	This is the sklearn GNB model. By passing in the train and test
	data, we can train the model and then test it. This function
	does exactly that and then returns the accuracy, as found
	with the function iter_accuracy
	'''
	def GNB_train_test_model(self, X_train, X_test, y_train, y_test):
		GNB_clf = GaussianNB()
		GNB_clf.fit(X_train, y_train)
		predicted = GNB_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	This is the sklearn Random Forest model. By passing 
	in the train and test data, we can train the model and then test it. 
	This function does exactly that and then returns the accuracy, as 
	found with the function iter_accuracy.
	'''
	def RF_train_test_model(self, X_train, X_test, y_train, y_test):
		RF_clf = RandomForestClassifier(n_estimators = self.RFEstimators)
		RF_clf.fit(X_train, y_train)
		predicted = RF_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)


	'''
	This is the sklearn Logistic Regression model. By passing 
	in the train and test data, we can train the model and then test it. 
	This function does exactly that and then returns the accuracy, as 
	found with the function iter_accuracy.
	'''
	def LOG_train_test_model(self, X_train, X_test, y_train, y_test):
		LOG_clf = LogisticRegression()
		LOG_clf.fit(X_train, y_train)
		predicted = LOG_clf.predict(X_test)
		actual = y_test

		return self.evaluatePerformance(actual,predicted)

	'''
	returns accuracy of the sample
	'''
	def evaluatePerformance(self, actual, predicted):
		accuracy = (actual == predicted).value_counts().get(True,0) / actual.size
		SSE_vals = (actual - predicted)**2
		SSE = SSE_vals.sum()

		return (accuracy, SSE)

	
	'''
	Main engine of the model. For each model specified above, this 
	function will run it and store the accuracy data as a dictionary.
	The keys for this dictionary are the names of the functions above,
	before the first underscore. This function allows you to specify
	the number of samples you would like to collect, the test ratio for
	how much of the dataset you want to predict, and a list of the
	models that you want to provide. For this model we are going to predict
	all five.
	'''
	def model_engine(self,classifiers):

		# Results dictionary
	    results_dict = {}

	    # Check all of classifiers
	    for classifier in classifiers:
	        res_list = []
	        model_tag = classifier.__name__.rsplit('_')[0]
	        print(model_tag)

	        if self.SpecificModel is not None and self.SpecificModel != model_tag:
	        	continue

	        for j in range(self.MonteCarloSampSize):
	        	print(j)

	        	#Start time for model
	        	startTime = dt.datetime.utcnow()

	        	#Get train and test variables
	        	X_train, X_test, y_train, y_test = train_test_split(self.Sample.iloc[:,:-1],self.Sample.iloc[:,-1],test_size = self.TestRatio)

	        	#Get performance and other metadata

	        	self.TrainRowNum = len(X_train)
	        	self.TestRowNum = len(X_test)
	        	self.ModelPerf = classifier(X_train,X_test,y_train,y_test)
	        	self.ModelName = model_tag

	        	#Add record to results DF
	        	res_list.append(self.ModelPerf)

	        	self.ModelDurationSec = (dt.datetime.utcnow() - startTime).total_seconds()

	        	#Add record to Logger
	        	self.Log.addResultRecord(self)

	        # Add results to results dict
	        results_dict[model_tag] = res_list

	        #Save Results Log
	        self.Log.saveResultsLog(self.ResLogFilename)

	    #Format and store the average results
	    self.resultsDF = pd.DataFrame.from_dict(results_dict)
	    averageResults = self.resultsDF.apply(lambda col: tuple(map(np.mean, zip(*col))),axis = 0).to_dict()

	    #Store average results for each model (order does not matter because of keys)
	    self.SVMPerf = averageResults.get('SVM',(0,0))
	    self.RFPerf = averageResults.get('RF',(0,0))
	    self.GNBPerf = averageResults.get('GNB',(0,0))
	    self.KNNPerf = averageResults.get('KNN',(0,0))
	    self.LOGPerf = averageResults.get('LOG',(0,0))


