import numpy as np
import matplotlib.pyplot as plt

from .gaussClassifier import *
from .gaussRegression import *

import warnings

"""
Utilities for Porous Ceramic Data Sets management

PCDataSet is a set of vectors. Each vector has the following shape, that should be matched by the csv data file :
25-100,100-200,200-400,Mixture Porogen volume %,"Porosité ouverte","Porosité fermée","Porosité totale",Fractured,Permeable,Extra_Label
(items can be empty)

Extra_Label is used to store extra information (eg. to mark fictive data)

Plus utilities to extract interesting data
"""

class PreparationParameters():
   """
   Initialize set of parameters
   """
   def __init__( self ):
      self.inputDim = 3          # Do we consider 200-400?
      self.outputFracture = True # Do we consider Fracture?
      self.outputPermea = True   # Do we consider Permeability?
      self.sumporogens = False   # Do we add the sum of porogens to the database?

class PCDataSet():

   """
   Initalize a dataset
   path : path of the csv file of the dataset. If none, then the dataset is empty
   """
   def __init__( self, path=None ):
      if path is None:
         self.multiVector = np.zeros( (0,10) )
      else:
         self.multiVector = np.genfromtxt(path, skip_header=1, delimiter=',')
         
      self.nbVectors = np.shape(self.multiVector)[0]
         
   """
   Adds data from a csv to total data heap (by concatenation)
   """
   def addFromCsv( self, path ):
      newdata = np.genfromtxt(path, skip_header=1, delimiter=',')
      if np.ndim(newdata) == 1: # If there is only one vector, transform it into a line matrix
         newdata = np.reshape( newdata, (1,10))
      
      self.multiVector = np.concatenate( (self.multiVector, newdata), axis=0 )
      self.nbVectors = np.shape(self.multiVector)[0]
   
   """
   Prepare data for classification
   """
   def prepareClassification( self, params=None ): # TODO: we need a way to defend against repeated points
   
      if params is None:
         params = PreparationParameters()
   
      # Remove nan values (put to zero)
      nantozero   = [0,1,2,9] # For those rows, nan is zero
      for j in nantozero:
         idnans = np.isnan( self.multiVector[:,j] )
         self.multiVector[idnans,j] = 0
         print("Completing database: "+str(np.sum(idnans))+" nans put to 0 in row "+str(j))
      
      # We can deduce Permeability from Fracture, and non-fracture from non-permeability
      nanFracture = np.isnan( self.multiVector[:,7] )
      nanPermeabl = np.isnan( self.multiVector[:,8] )
      isFract = self.multiVector[:,7] > .99 # Should be 1, but let space for round off errors
      isPerme = self.multiVector[:,8] > .99
      frImplPer = np.logical_and( np.logical_and( isFract, np.logical_not( nanFracture ) ), nanPermeabl ) # Fractured implies Permeable.
      perImplFr = np.logical_and( np.logical_and( np.logical_not(isPerme), np.logical_not( nanPermeabl ) ), nanFracture )# Non permeable implies Non Fractured
      self.multiVector[perImplFr,7] = False
      self.multiVector[frImplPer,8] = True
      
      # Remove nan values (suppress lines)
      nantoremove = list(range(params.inputDim)) # For those rows, nan is fatal: remove from database
      if params.outputFracture:# and not params.outputPermea:
         nantoremove.append(7)
      if params.outputPermea:# and not params.outputFracture:
         nantoremove.append(8)
         
      arevalid = np.ones( self.nbVectors, dtype=bool ) # Stores valid vectors
      for j in nantoremove:
         idnans = np.isnan( self.multiVector[:,j] )
         arevalid = np.logical_and(arevalid,np.logical_not(idnans))
         print("Cutting database: "+str(np.sum(idnans))+" vectors containing nans ignored in row "+str(j))
   
      # Remove values having non-zero quantities outside of inputDim
      for j in range(params.inputDim,3):
         iszero = (self.multiVector[:,j] == 0)
         arevalid = np.logical_and(arevalid,iszero)
         print("Cutting database: "+str(self.nbVectors-np.sum(iszero))+" vectors containing non-zero values in row "+str(j)+" ignored")
   
      self.nbValidVectors = np.sum(arevalid)
   
      # Then, extract specific data for classification
      self.inputVectors = 1/100 * self.multiVector[arevalid,0:params.inputDim] # Proportions of each type of porogen (and not percentage)
      
      if params.sumporogens:
         self.inputVectors = addSum( self.inputVectors )
      
      self.inputDim = np.shape(self.inputVectors)[1] # dataSet.inputDim is +1 if sumporogens is active
      
      # Finally, boolean class vectors
      fracture  = np.array( self.multiVector[arevalid,7], dtype=bool ) # Is the sample fractured?
      permeable = np.array( self.multiVector[arevalid,8], dtype=bool ) # Is the sample permeable?
      
      if params.outputFracture:
         if params.outputPermea:
            self.classVector = np.logical_and(permeable,np.logical_not(fracture)) # Sample has to be permeable and not fractured to be OK
         else:
            self.classVector = np.logical_not(fracture)               # Sample has to be not fractured to be OK
      else:
         if params.outputPermea:
            self.classVector = permeable                    # Sample has to be permeable to be OK
         else:
            warnings.warn("No criterion is avaliable to perform classification!")
            
      self.fictiveMarker = np.array( self.multiVector[arevalid,9], dtype=bool ) # Is the sample fictive?
      
      self.prepaParameters = params
            
   """
   Plot the database
   """
   def plot( self, figName=None, show=True, subPlot=None ):
   
      # First, decide the colour of each point
      notClassVector = np.logical_not( self.classVector )
      notFictive     = np.logical_not( self.fictiveMarker )
      
      redCrosses   = np.logical_and( notClassVector, notFictive )
      blackCrosses = np.logical_and( self.classVector, notFictive )
      redCircles   = np.logical_and( notClassVector, self.fictiveMarker )
      blackCircles = np.logical_and( self.classVector, self.fictiveMarker )
   
      # TODO: if dim==1
   
      if figName is None:
         figName = plt.figure()
   
      if self.prepaParameters.inputDim == 2:
         plt.scatter(self.inputVectors[redCrosses,0],self.inputVectors[redCrosses,1],marker='+',c='red')
         plt.scatter(self.inputVectors[blackCrosses,0],self.inputVectors[blackCrosses,1],marker='+',c='blue')
         plt.scatter(self.inputVectors[redCircles,0],self.inputVectors[redCircles,1],marker='o',c='red')
         plt.scatter(self.inputVectors[blackCircles,0],self.inputVectors[blackCircles,1],marker='o',c='blue') # Blue is the new black
         ax = plt.gca()
         ax.set_aspect('equal')
         plt.xlabel('25-100')
         plt.ylabel('100-200')
         
         if show:
            plt.show()
            
      elif self.prepaParameters.inputDim == 3:
         if subPlot is None:
            subPlot = figName.add_subplot(projection='3d') # TODO: manage without this subplot
            
         subPlot.scatter(self.inputVectors[redCrosses,0],self.inputVectors[redCrosses,1],self.inputVectors[redCrosses,2],marker='+',c='red')
         subPlot.scatter(self.inputVectors[blackCrosses,0],self.inputVectors[blackCrosses,1],self.inputVectors[blackCrosses,2],marker='+',c='blue')
         subPlot.scatter(self.inputVectors[redCircles,0],self.inputVectors[redCircles,1],self.inputVectors[redCircles,2],marker='o',c='red')
         subPlot.scatter(self.inputVectors[blackCircles,0],self.inputVectors[blackCircles,1],self.inputVectors[blackCircles,2],marker='o',c='blue') # Blue is the new black
         subPlot.set_aspect('equal')
         subPlot.set_xlabel('25-100')
         subPlot.set_ylabel('100-200')
         subPlot.set_zlabel('200-400')
      
      else:
         warnings.warn("no plotting procedure avaliable for this dimension")
         
      if show:
            plt.show()
   
   """
   Initialize data format and regression stuff
   """
   def initializeClassification( self, paramsC=None, paramsR=None, classifier=None ):
      if classifier is None: # TODO: and self.classifier is None
         self.classifier = gaussClassifier( paramsC, paramsR )
      else:
         self.classifier = classifier
      
      if not hasattr(self.classifier.regression.params,"corrlenVect"): # Impose initial correlation lenght vector if does not exist yet
         self.classifier.regression.isotropic_corrlen( self ) # Handle anisotropy
   
   """
   Classify the data, with a set of parameters for the Classification and one for the underlying regression
   """
   def classify( self, queryPts, paramsC=None, paramsR=None, classifier=None ): # TODO: this should call initializeClassification (but handle the isotropic_corrlen properly)
   
      # First we need to handle query points
      self.queryPts = queryPts
      self.nbQuery = np.shape(self.queryPts)[0]
      if self.prepaParameters.sumporogens: # Add the sum as extra component
         self.queryPts = addSum( self.queryPts )
      
      self.classifier.classify( self )
      
   """
   Optimizes the classifier's hyperparameters via Log Marginal Likelihoodof the Regression
   """
   def optimizeLogMLClassif( self, paramsC=None, paramsR=None, classifier=None ): # TODO: this should call initializeClassification (but handle the isotropic_corrlen properly)
      if classifier is None:
         self.classifier = gaussClassifier( paramsC, paramsR )
      else:
         self.classifier = classifier

      self.classifier.optimizeLogML( self )
      
      
"""
Stand-alone utility functions
"""

"""
Compute the sum of components in each vector of the database (argument sumporogens)
"""
def addSum( multivect ):
   #nvect = np.shape(multivect)[0]
   sumv = np.sum( multivect, axis=1 )
   sumv = sumv.reshape( (len(sumv),1) ) # Transform Vector into one-row matrix
   return np.concatenate( ( multivect, sumv ), axis=1  )
   
   
if __name__ == "__main__":
   data = PCDataSet('../samples/initial_samples.csv')
   data.addFromCsv('../samples/fictive_samples.csv')
  
   #print(data.multiVector)
   #print(np.shape(data.multiVector))
  
   params = PreparationParameters()
   #params.outputPermea = False
  
   data.prepareClassification(params)
   
   #print(np.shape(data.inputVectors))
   print(data.inputVectors)
   print(data.classVector)
   print(data.fictiveMarker)


