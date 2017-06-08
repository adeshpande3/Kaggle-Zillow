import pandas as pd 
import numpy as np 
import csv
import sklearn
from sklearn import linear_model
from cleanData import cleanData # Function in cleanData.py
from cleanData import createTrainingMatrices # Function in cleanData.py

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

# Train the model
print 'Training the model'
model = linear_model.LinearRegression()
model.fit(xTrain, yTrain)

# Get the predictions
print 'Test prediction'
preds = model.predict(properties);

# Getting submission ready
print 'Getting submission ready'
numTestExamples = properties.shape[0]
numPredictionColumns = 7
predictions = []
for index, pred in enumerate(preds):
	parcelNum = int(cleanedPropertyData[index][0])
	predictions.append([parcelNum,pred,pred,pred,pred,pred,pred])

print 'Writing results to CSV'
firstRow = [['ParcelId', '201610', '201611', '201612', '201710', '201711', '201712']]
with open("preds.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(firstRow)
    writer.writerows(predictions)






