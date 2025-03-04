import numpy as np
#import matplotlib.pyplot as plt

from dataSet import *
from gaussRegression import *

"""
Utilities for active learning.
"""

class learnOptParams():
   """
   Initialize a set of parameters for Active Learning
   """
   def __init__( self ):
      self.npts         = 5     # Number of requested points
      self.nptsTar      = 5     # Target number of points (only used for computing corrlen if useRegsKer == False)
      self.niter        = 100   # Number of gradient iterations per nb of points
      self.ponderation  = 1.    # Ponderation parameter between similarity and uncertainty
      self.analyticDiff = True  # Should we do analytic differentiation or variationnal?
      
      self.useRegsKer = False          # Should we use regression kernel?
      self.corrlen    = 1              # Correlation length of optimizer's own kernel
      self.std        = 1              # Standard deviation of optimizer's own kernel
      self.kernelT    = "exponential"  # Kernel type


class learningOptimizer():
   """
   Initialize stuff
   """
   def __init__( self, regression, dataSet, params=None ):
   
      if params is None:
         params = learnOptParams()
   
      self.regression  = regression  # Associate to a regression problem
      self.dataSet     = dataSet     # Associate to a data set
      self.params      = params      # Associate to a set of parameters
      
      if not self.params.useRegsKer: # Determinate best correlation length (in case its not led by the regression kernel)
         # Average volume per point is equal to the volume of the tetrahedron divided by nb of point
         # And correlation length is equal to the d-th root of this volume (with d the dimension)
         self.params.corrlen = (.5**(1/(dataSet.inputDim-1)) / (dataSet.nbValidVectors+self.params.nptsTar))**(1/dataSet.inputDim) # .5 because we work in a tetraedron and not a cube
         # Just for reference: dataSet.inputDim is dimension+1 if sumporogens is active.
         print( "correlation length for composite cost function: " + str(self.params.corrlen) )
      
      if self.params.useRegsKer:
         self.params.corrlenVect = self.regression.params.corrlenVect
      else:
         self.params.corrlenVect = self.params.corrlen * np.ones( self.dataSet.inputDim ) # Update vectorial correlation length.
      
      minL = min( np.min(self.params.corrlenVect) , np.min(self.regression.params.corrlenVect) ) # Minimal correlation length
      self.mergeRadius2 = (.01 * minL)**2 # Radius for two points to be the same
      
   """
   Compute elements for the composite cost function
   """
   def computeCFunctionElements( self, samples ):
      if self.dataSet.prepaParameters.sumporogens: # Add the sum as extra component.
         samples = addSum( samples )
   
      self.regression.regress( self.dataSet, queryPts=samples )
      
      if self.params.useRegsKer:
         kernelData = self.regression.params
      else: # Copy optimizer's own data into the dataset for covariance computation
         kernelData = copy(self.regression.params)
         kernelData.std         = self.params.std
         kernelData.corrlenVect = self.params.corrlenVect
         kernelData.kernelT     = self.params.kernelT
      
      C1 = self.regression.computeCCovariance( self.dataSet.inputVectors, samples, kernelData )
      C2 = self.regression.computeCCovariance( samples, samples, kernelData )
      
      return C1, C2
      
   """
   Compute composite cost function for a given cloud of points
   """
   def computeCFunction( self, samples ):
      C1, C2 = self.computeCFunctionElements( samples ) # Pre-compute needed elements
      phixi = np.sum( np.square(self.regression.regsOutput) ) # First term: sum of mean values at points
      phik1 = np.sum( np.square(C1) ) # Second term: similarity with the points in the data base (Frobenius norm of Covariance)
      phik2 = np.sum( np.square(C2) ) # Third term: similarity between the points (Frobenius norm of Covariance)
      return phixi + self.params.ponderation * ( phik1 + phik2 )
      
   """
   Compute contribution of each point to the composite cost function
   """
   def computeCFunctionContrib( self, samples, detail=False ):
      C1, C2 = self.computeCFunctionElements( samples ) # Pre-compute needed elements

      # The contributions are computed as sums over columns. This is not really true for phik2. TODO: see
      phixi = np.square(self.regression.regsOutput).flatten() # First term: sum of mean values at points
      phik1 = np.sum( np.square(C1), axis=0 ) # Second term: similarity with the points in the data base
      phik2 = np.sum( np.square(C2), axis=0 ) # Third term: similarity between the points
      
      if detail: # Give the details of the terms
         return phixi + self.params.ponderation * ( phik1 + phik2 ), phixi, phik1, phik2
      else:
         return phixi + self.params.ponderation * ( phik1 + phik2 )
      
   """
   Compute derivative of the composite cost function by forward finite differences
   This function also returns the value of the cost function
   """
   def computeDCFFFD( self, samples ):
      
      step = .001 * np.min(self.regression.params.corrlenVect) # Finite differences step
      nsamples, ndim = np.shape(samples)
      phi0 = self.computeCFunction( samples ) # Evaluate cost function at the current value
      Dphi = np.zeros( (nsamples,ndim) )
      
      for j in range(ndim):
         for i in range(nsamples):
            sampleij = np.copy(samples)
            sampleij[i,j] += step
            phiij = self.computeCFunction( sampleij )
            Dphi[i,j] = (phiij-phi0) / step
            
      return Dphi, phi0
      
   """
   Compute derivative of the composite cost function by centered finite differences
   This function is approx. twice more expensive as computeDCFFFD, but necessary for non-differentiable points
   """
   def computeDCFCFD( self, samples ):
      
      step = .01 * np.min(self.regression.params.corrlenVect) # Finite differences step
      nsamples, ndim = np.shape(samples)
      Dphi = np.zeros( (nsamples,ndim) )
      
      for j in range(ndim):
         for i in range(nsamples):
            sampleijp = np.copy(samples)
            sampleijp[i,j] += step
            sampleijm = np.copy(samples)
            sampleijm[i,j] -= step
            
            phiijp = self.computeCFunction( sampleijp )
            phiijm = self.computeCFunction( sampleijm )
            
            Dphi[i,j] = (phiijp-phiijm) / (2*step)
            
      return Dphi
      
   """
   Compute analytical derivative of composite cost function
   """
   def computeDCFA( self, samples ):
   
      if self.dataSet.prepaParameters.sumporogens: # Add the sum as extra component.
         samples = addSum( samples )

      nsamples, ndim = np.shape(samples)
      dphixi = np.zeros( (nsamples,ndim) )
      dphik1 = np.zeros( (nsamples,ndim) )
      dphik2 = np.zeros( (nsamples,ndim) )

      self.regression.regress( self.dataSet, queryPts=samples ) # Regression value
      Dregs = self.regression.Dregress( self.dataSet, queryPts=samples ) # Derivatives of the regression value
      
      for i in range(ndim):
         dphixi[:,i] = 2 * self.regression.regsOutput * Dregs[:,i] # Derivative of the sum of squared.
         
      if self.params.useRegsKer:
         kernelData = self.regression.params
      else: # Copy optimizer's own data into the dataset for covariance computation
         kernelData = copy(self.regression.params)
         kernelData.std         = self.params.std
         kernelData.corrlenVect = self.params.corrlenVect
         kernelData.kernelT     = self.params.kernelT
         
      C1  = self.regression.computeCCovariance( self.dataSet.inputVectors, samples, kernelData )
      C2  = self.regression.computeCCovariance( samples, samples, kernelData )
      DC1 = self.regression.computeDCCovariance( self.dataSet.inputVectors, samples, kernelData )  # Derivatives of the covariances matrices
      DC2 = self.regression.computeDCCovariance( samples, samples, kernelData )
      
      for i in range(ndim):
         dphik1[:,i] = 2*np.sum( DC1[:,:,i]*C1, axis=0 ) # Correlation with the data.
         dphik2[:,i] = 4*np.sum( DC2[:,:,i]*C2, axis=0 ) # Autocorrelation: samples appear 2 times.
      
      derivative = dphixi + self.params.ponderation * ( dphik1 + dphik2 )
      
      # Time for sumporogens to make yet again our lives a small bit harder
      if self.dataSet.prepaParameters.sumporogens: # All dimensions continbute to the sum
         for i in range(ndim-1):
            derivative[:,i] += derivative[:,-1]
         derivative = derivative[:,:-1]
      
      return derivative
      
   """
   gradient optimization for the composite cost function
   FD0: should we use centered finite differences at iteration 1 (in case cost function is not differentiable at initial point)
   """
   def gradientOptim( self, samples0, step=1, FD0=False, niter=100, analyticDiff=False, stagCrit = 1e-6 ):
      
      samples = np.copy(samples0)
      
      firstConverge = True # A flag to detect first convergence
      stepFirst = step # This will store the step at first "right" iterate
      
      for i in range(niter):

         if FD0 and i==0: # For first iteration, use Centered Finite Differences (because the cost function might be not differentiable analytically for exponential kernel)
            Dphi = self.computeDCFCFD( samples )
            phi0 = self.computeCFunction( samples )
         elif analyticDiff: # Compute the derivative analytically. TODO: understand where it can be optimized further...
            Dphi = self.computeDCFA( samples )      # TODO: see if by chance that function can also compute phi0 at low cost
            phi0 = self.computeCFunction( samples )
            """# DEBUG
            Dphip = self.computeDCFCFD( samples )
            print(np.linalg.norm(Dphi,'fro'))
            print(np.linalg.norm(Dphip,'fro'))
            print(np.linalg.norm(Dphi-Dphip,'fro'))"""
         else: # Use Forwards Finite Differences
            Dphi, phi0 = self.computeDCFFFD( samples ) 
         
         """#DEBUG
         if i==niter-1:
            print(samples)
            print(Dphi)
         #END DEBUG"""
                    
         for j in range(100): # Linesearch
            step = step/2
            delta = -step*Dphi
            samplesT = samples + delta
            samplesT = self.projectTetraedron( samplesT ) # Minimization under constraint
            phiT = self.computeCFunction( samplesT )

            if phiT < phi0: # Cost function decreases: stop decreasing step
               break
            
         if phiT >= phi0: # Linesearch did fail: we cannot optimize further.
            break
         else: # Working right: store step
            if firstConverge:
               stepFirst = step  # Store first step
            firstConverge = False
         
         # TODO: re-use phiT as next phi0 to save one computation

         samplesO = np.copy(samples)  # Remember old samples
         samples  = np.copy(samplesT)
         step = 4*step # Increase for next iteration
         
         if np.max(np.abs(samples-samplesO)) < stagCrit: # Stagnation criterion met. TODO: it is actually quasi-never the case...
            break
         
      return samples, stepFirst, step # Return solution, first step and last step
   
   """
   Compute the best set of points wrt. the composite cost function of uncertainty and similarity
   """
   def optimBestCF( self, samples0=None ):
   
      if samples0 is None:
         samples0 = np.copy(self.dataSet.inputVectors) # Initialization
         if self.dataSet.prepaParameters.sumporogens: # We need to remove added value
            samples0 = samples0[:,:-1]
   
      nsamples = np.shape(samples0)[0]
      
      if nsamples <= self.params.npts:
         warnings.warn("Algorithm initializing with less samples than required.")
      
      samples, stepFirst, _ = self.gradientOptim( samples0, step=1e-3, FD0=True, niter=self.params.niter, analyticDiff=self.params.analyticDiff ) # FD0 means that at first iteration we use Centered finite differences
      
      while nsamples > self.params.npts:
         print(np.shape(samples))
         
         # Move the samples
         samples, stepFirst, _ = self.gradientOptim( samples, step=1e-3, niter=self.params.niter, analyticDiff=self.params.analyticDiff )
         samples = self.mergeSimilar( samples ) # Merge values that are too close. TODO: understand why that happends...
         # Remove the point that contributes the most to the cost function
         contrib = self.computeCFunctionContrib( samples )
         killyo = np.argmax(contrib)    # Kill the maximal point
         samples = np.delete(samples, killyo, 0)
         
         nsamples = np.shape(samples)[0]
         #samples += .1*minL * ( 2*np.random.rand(nsamples,np.shape(samples)[1]) - 1. ) # Randomly perturbate the points (to exit local maxima and such)
         
      
      # Finally: iterate a bit more and merge similars
      samples, stepFirst, _ = self.gradientOptim( samples, step=1e-3, niter=self.params.niter, analyticDiff=self.params.analyticDiff )
      samples = self.mergeSimilar( samples )
      samples, stepFirst, _ = self.gradientOptim( samples, step=1e-3, niter=self.params.niter, analyticDiff=self.params.analyticDiff )
      self.preferredSamples = samples

      contribData = self.computeCFunctionContrib( samples, detail=True )
      self.contrib2CF = contribData[0] # Contribution of samples to cost function
      self.contrib2Xi = contribData[1] # Contribution of samples to uncertainty term
   
   """
   Duplicate the best points and re-compute the best set
   """
   def duplicateNRecompute( self, nb=5 ):
   
      if self.preferredSamples is None:
         warnings.warn("Learning Optimizer has no preferred samples, it cannot duplicate them.")
         return
         
      if (nb > np.shape(self.preferredSamples)[0]):
         warnings.warn("Learning Optimizer has not enough preferred samples for the required duplication number.")
         nb = np.shape(self.preferredSamples)[0]
         
      # Re-compute contributions to cost function in case something changed.
      self.contrib2CF = self.computeCFunctionContrib( self.preferredSamples )
      
      # Rank the samples by their contribution to the cost function
      ind     = np.argsort(self.contrib2CF)
      samList = self.preferredSamples[ind,:]
      samList = samList[:nb,:]
      dim = np.shape(samList)[1]
      
      # Create the list of duplicated points
      ndup = 2*dim*nb # Nb of added points
      dupList = np.zeros( (ndup,dim) )

      for i in range(nb):
         for j in range(dim):
            ind = 2*(i+nb*j)
            dupList[ind,:]   = samList[i,:]
            dupList[ind,j]   = dupList[ind,j] + 2*self.mergeRadius2
            dupList[ind+1,:] = samList[i,:]
            dupList[ind+1,j] = dupList[ind,j] - 2*self.mergeRadius2
      
      samples0 = np.concatenate((samList,dupList)) # Concatenate lists
      
      self.optimBestCF( samples0=samples0 ) # Run the optimization
   
   """
   Merge similar points
   """
   def mergeSimilar( self, samples ):
   
      nsamples = np.shape(samples)[0]
      
      CrossD2 = CrossDist2( samples, samples )
      toremove = []
      for j in range(nsamples-1):
         myline = CrossD2[j,j+1:]
         if np.sum(myline<self.mergeRadius2) > 0: # There exists an other point really close: remove that one
            toremove.append(j)
      
      samples = np.delete(samples, toremove, 0)
      return samples
   
   """
   Is the set of points inside the unitary tetraedron?
   """
   def insideTetraedron( self, samples ):
   
      if np.sum( samples<0. ) > 0: # There are negative values
         return False
         
      sums = np.sum( samples, axis=1 )
      
      if np.sum( sums>1. ) > 0: # There are total values outside the tetrahedron
         return False
         
      return True
      
   """
   Project a set of points in the unitary tetraedon
   """
   def projectTetraedron( self, samples ): # TODO: get that function outside of the object
   
      nsamples = np.shape(samples)[0]
      sD = np.ones((np.shape(samples)[1],1)) # Sommation over the dimensions
      sS = np.ones((1,nsamples))             # Sommation over the samples
      sDsD = np.shape(samples)[1]            # Number of samples. TODO: check it is right
      
      samT = samples.T # On my notebook, I have the formulaes for columns vectors :P
      
      # Step 1 : projection on the diagonal hyperplane. TODO: see if this step is needed at all
      sums = np.sum( samples, axis=1 )
      superior =  sums>1.
      samT[:,superior] = samT[:,superior] - (sD @ ((sD.T @ samT[:,superior])/sDsD)) * sD + (sD @ sS[:,superior])/sDsD # x <- x - (sD.Tx)/(sD.TsD) sD + sD/(sD.TsD)
      
      # Step 2 : projection on the lower hyperplanes
      inferior = samT<0.
      samT[inferior] = 0.
      
      # Step 3 : projection on the diagonal hyperplane while ensuring coefficients [inferior] stay zero
      sDt = sD @ sS
      sDt[inferior] = 0
      sums = np.sum( samples, axis=1 )
      superior =  sums>1.
      for i in range(nsamples):
         if not superior[i]:
            continue  # Move on to next one if no projection is needed
            
         sDi = sDt[:,i]
         sDsDi = np.sum(np.square(sDi))
         samT[:,i] = samT[:,i] - ((sDi.T @ samT[:,i])/sDsDi) * sDi + sDi/sDsDi
         
      return samT.T
         
      
