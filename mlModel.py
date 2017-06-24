import pandas as pd 
import numpy as np 
import csv
import sklearn
from utils import cleanData 
from utils import createTrainingMatrices 
from utils import createKaggleSubmission
from utils import findBestMLModel
import os.path
from sklearn import linear_model

## Dataset Cleaning TODOs ##
# How do you save a dataframe in Numpy

if (os.path.isfile('xTrain.npy') and os.path.isfile('yTrain.npy') and os.path.isfile('properties.npy')):
	print 'Loading in precomputed xTrain, yTrain, and properties'
	xTrain = np.load('xTrain.npy')
	yTrain = np.load('yTrain.npy')
	propertiesDataFrame = pd.read_csv('properties_2016.csv', low_memory=False)
	properties = np.load('properties.npy')
	cleanedPropertyData = pd.np.array(properties)
else:
	# Load in the data
	print 'Loading in data'
	propertiesDataFrame = pd.read_csv('properties_2016.csv', low_memory=False)
	trainDataFrame = pd.read_csv('train_2016.csv')
	sampleSubDataFrame = pd.read_csv('sample_submission.csv')

	# Clean the data
	print 'Cleaning the data'
	properties = cleanData(propertiesDataFrame)
	cleanedPropertyData = pd.np.array(properties)
	print 'Shape of the cleaned data matrix:', cleanedPropertyData.shape

	print 'Computing xTrain and yTrain'
	xTrain, yTrain = createTrainingMatrices(properties, trainDataFrame)
	print 'Shape of the xTrain matrix:', xTrain.shape
	print 'Shape of the yTrain matrix:', yTrain.shape
	np.save('xTrain', xTrain)
	np.save('yTrain', yTrain)
	np.save('properties', properties)

# Train the model
#print 'Finding the best model'
#model = findBestMLModel(xTrain, yTrain)
#print 'The best model is:', model
print 'Training the model'
model = linear_model.Lasso()
model.fit(xTrain, yTrain)

createKaggleSubmission(model, properties, cleanedPropertyData)






