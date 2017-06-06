import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
from cleanData import cleanData # Function in cleanData.py

# Load in the data
print 'Loading in data'
propertiesDataFrame = pd.read_csv('properties_2016.csv', low_memory=False)
trainDataFrame = pd.read_csv('train_2016.csv')
sampleSubDataFrame = pd.read_csv('sample_submission.csv')

# Clean the data
print 'Cleaning the data'
properties = cleanData(propertiesDataFrame)
cleanedPropertyData = pd.np.array(properties)
print cleanedPropertyData.shape

# Train the model

# Get the predictions





