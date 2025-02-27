import numpy as np

from gaussRegression import *

"""
Utilities for Gaussian Processes classification
"""

class gaussClassifierParams():
   """
   Initializes a set of parameters for the Gaussian Classifier
   """
   def __init__( self ):
      self.pseudoinf = 10 # Value taken as infinity

class gaussClassifier():
   """
   Initialize a classifier.
   It has two sets of parameters: one for classification and one for the underlying regression
   """
   def __init__( self, paramsC=None, paramsR=None ):
      if paramsC is None:
         paramsC = gaussClassifierParams()
         
      if paramsR is None:
         paramsR = gaussRegressionParams()
   
      self.params = paramsC
      self.regression = gaussRegression( paramsR )
      
   """
   Computes Suitable Regression vector to transform classification problem into a regression problem
   """
   def getRegsVector( self, dataSet ):
      regsVector = np.zeros( (dataSet.nbValidVectors,1) )
      regsVector[dataSet.classVector,0]  = self.params.pseudoinf   # True  -> +pseudoinf
      regsVector[np.logical_not(dataSet.classVector),0] = -self.params.pseudoinf  # False -> -pseudoinf
      
      return regsVector
      
   """
   Perform classification
   Basically, just transforms the classification problem into a regression problem
   """
   def classify( self, dataSet ):
   
      if not hasattr(dataSet,"regsVector"): # Compute suitable regression vector if needed
         dataSet.regsVector = self.getRegsVector( dataSet )
      
      self.regression.regress( dataSet ) # Perform Gaussian Process Regression
      
      self.classOutput = 1 / (1 + np.exp(-self.regression.regsOutput)) # Term-to-term sigmoid. TODO: handle overflow
      
   """
   Optimize Regression hyperparameters
   """
   def optimizeLogML( self, dataSet ):
   
      if not hasattr(dataSet,"regsVector"): # Compute suitable regression vector if needed
         dataSet.regsVector = self.getRegsVector( dataSet )
   
      self.regression.optimizeLogML( dataSet )
   
   
