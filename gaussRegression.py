from copy import copy

import numpy as np
import matplotlib.pyplot as plt

import warnings

"""
Utilities for Gaussian processes regression (or Kriging)
"""

class gaussRegressionParams():
   """
   Initializes a set of parameters for the Gaussian Regression
   """
   def __init__( self ):
      self.std                = 1 # Standard deviation of the output
      self.stdM               = 1 # Standard deviation of the measurement (0 means interpolating Kriging)
      self.corrlen            = 1 # Correlation length
      self.kernelT            = "gaussian" # Kernel type
      self.anisotropy         = False # Anisotropic kernel. If true, when optimizing, correlation lengths will be computed separately
      
      self.minAnisotropyRatio = 0 # Minimal ratio between the smallest and biggest characteristic lengths
      self.noptimIter         = 100 # Nb of iterations for optimization algo
      self.doBFGS             = False # Wether to use BFGS instead of plain gradient algo
      self.kernelI            = None # It is possible to initialize the optimozation with an other computation with an other type of kernel.

class gaussRegression():
   """
   Initalizes the Gaussian Regression model
   """
   def __init__( self, params=None ):
      if params is None:
         params = gaussRegressionParams()
      
      self.params = params
      
   """
   Initializes isotropic correlation lenght as a vector
   """
   def isotropic_corrlen( self, dataSet, corrlen=None ):
      if corrlen is None:
         corrlen = self.params.corrlen
      else:
         self.params.corrlen = corrlen # If corrlen is given, it overrides the one stored in parameter.
         
      self.params.corrlenVect = corrlen * np.ones( dataSet.inputDim ) # TODO: when should this be used?
      
   """
   Compute Cross-Covariance matrix for two sets of vectors
   """
   def computeCCovariance( self, u1, u2, paramSet ):
   
      crossDist2 = CrossDist2( u1, u2, paramSet.corrlenVect ) # Ponderated Cross-distance on the database
      if paramSet.kernelT == "gaussian":
         CCov = paramSet.std**2 * np.exp( -1/2 * crossDist2 ) # Term-to-term product
      elif paramSet.kernelT == "exponential":
         CCov = paramSet.std**2 * np.exp( -np.sqrt(crossDist2) )
      else:
         warnings.warn("Kernel type not recognized")
         
      return CCov
      
   """
   Compute Derivative of Cross Covariance matrix wrt. a vector
   Result is a tensor of order 3 : cross-covariance + dimension
   """
   def computeDCCovariance( self, u1, u2, paramSet ):
      diff = CrossDiff( u1, u2, paramSet.corrlenVect )
      crossDist2 = np.sum( np.square(diff), axis=2 ) # Ponderated squared Cross-distance on the database
      
      dim = np.shape(diff)[2]
      DCov = np.zeros( np.shape(diff) ) # Tensor of dimension (s1,s2,dim)
      
      if paramSet.kernelT == "gaussian":
         for i in range(dim):
            DCov[:,:,i] = - paramSet.std**2 * diff[:,:,i] * np.exp( -1/2 * crossDist2 ) / paramSet.corrlenVect[i]
      elif paramSet.kernelT == "exponential":
         dist = np.sqrt(crossDist2)
         maxdist = np.max(dist)
         dist = np.maximum( dist, 1e-6*maxdist ) # I do not want to divide by 0
         for i in range(dim):
            DCov[:,:,i] = paramSet.std**2 * diff[:,:,i] * np.exp( -dist ) / dist / paramSet.corrlenVect[i]
      else:
         warnings.warn("Kernel type not recognized")
      
      return DCov
      
   """
   Compute Covariance matrix for the data
   """
   def computeCovariance( self, dataSet, paramSet ):
      Cov = self.computeCCovariance( dataSet.inputVectors, dataSet.inputVectors, paramSet )
      return Cov + paramSet.stdM**2 * np.identity( dataSet.nbValidVectors ) # Add diagonal term for measurement error (non-interpolating GP)
      
   """
   Compute Covariance matrix for the data AND estimate its norm.
   """
   def computeCovarianceNNorm( self, dataSet, paramSet ):
      Cov = self.computeCCovariance( dataSet.inputVectors, dataSet.inputVectors, paramSet )
      froNorm = np.linalg.norm(Cov,'fro')/np.sqrt(dataSet.nbValidVectors)
      return Cov + paramSet.stdM**2 * np.identity( dataSet.nbValidVectors ), froNorm # Add diagonal term for measurement error (non-interpolating GP)

   """
   Computes Log Marginal Likelyhood
   """
   def computeLogML( self, dataSet, paramSet, CovM ):
      Cm1y = np.linalg.solve( CovM, dataSet.regsVector )
      logML = - dataSet.nbValidVectors/2*np.log(2*np.pi) \
              - dataSet.nbValidVectors/2*np.log(np.linalg.det(CovM)) \
              - .5* dataSet.regsVector.T @ Cm1y                                 # Log Marginal likelyhood
      return logML.item() # item because at this point its a 1x1 array

   """
   Computes derivative of Log Marginal Likelyhood
   """
   def computeDLogML( self, dataSet, paramSet, CovM ):

      # Compute elementary matrix derivatives
      dAstdM = 2*paramSet.stdM * np.identity(dataSet.nbValidVectors) # Derivative of A wrt. measurement standard deviation
      
      crossDist2 = CrossDist2( dataSet.inputVectors, dataSet.inputVectors, paramSet.corrlenVect ) # Ponderated Cross-distance on the database
      if paramSet.kernelT == "gaussian":  
         if paramSet.anisotropy:
            dAcorl = np.zeros( (dataSet.nbValidVectors,dataSet.nbValidVectors,dataSet.inputDim) )
            for i in range(dataSet.inputDim):
               v1 = dataSet.inputVectors[:,i]
               v1 = v1.reshape( (len(v1),1) ) # Only coordinate i
               v2 = paramSet.corrlenVect[i]
               v2 = np.atleast_1d(v2) #v2 = v2.reshape( 1 )
               crossDist2i = CrossDist2( v1, v1, v2 ) # Distance in direction i. TODO: efficiency
               dAcorl[:,:,i] = paramSet.std**2/paramSet.corrlenVect[i] * crossDist2i * np.exp( -1/2 * crossDist2 )
         else:
            dAcorl = paramSet.std**2/paramSet.corrlen * crossDist2 * np.exp( -1/2 * crossDist2 ) # Derivative of A wrt. correlaton lenght
            
      elif paramSet.kernelT == "exponential":
         if paramSet.anisotropy:
            dAcorl = np.zeros( (dataSet.nbValidVectors,dataSet.nbValidVectors,dataSet.inputDim) )
            for i in range(dataSet.inputDim):
               v1 = dataSet.inputVectors[:,i]
               v1 = v1.reshape( (len(v1),1) ) # Only coordinate i
               v2 = paramSet.corrlenVect[i]
               v2 = np.atleast_1d(v2) #v2 = v2.reshape( 1 )
               crossDist2i = CrossDist2( v1, v1, v2 ) # Distance in direction i. TODO: efficiency
               dAcorl[:,:,i] = paramSet.std**2/paramSet.corrlenVect[i] * np.sqrt(crossDist2i) * np.exp( -np.sqrt(crossDist2) )
         else:
            dAcorl = paramSet.std**2/paramSet.corrlen * np.sqrt(crossDist2) * np.exp( -np.sqrt(crossDist2) ) # Derivative of A wrt. correlaton lenght

      else:
         warnings.warn("Kernel type not recognized")
      
      # Remark : no derivative wrt. std because it has the same role as stdM
      Am1dAstdM = np.linalg.solve( CovM, dAstdM ) # A^{-1} dAstdM (the trace of this is the derivative of the logairthm of det(A))
      
      if paramSet.anisotropy:
         Am1dAcorl = np.zeros( (dataSet.nbValidVectors,dataSet.nbValidVectors,dataSet.inputDim) )
         for i in range(dataSet.inputDim):
            Am1dAcorl[:,:,i] = np.linalg.solve( CovM, dAcorl[:,:,i] )
      else:
         Am1dAcorl = np.linalg.solve( CovM, dAcorl ) # A^{-1} dAcorl (the trace of this is the derivative of the logairthm of det(A))
      
      # Compute derivative of the inverse
      Cm1y = np.linalg.solve( CovM, dataSet.regsVector )
      
      dlogMLstdM = - dataSet.nbValidVectors/2 * np.trace( Am1dAstdM ) + .5 * Cm1y.T @ ( dAstdM @ Cm1y )
      
      if paramSet.anisotropy:
         dlogMLcorl = np.zeros( dataSet.inputDim )
         for i in range(dataSet.inputDim): # TODO: factorize all those "for i"
            it = - dataSet.nbValidVectors/2 * np.trace( Am1dAcorl[:,:,i] ) + .5 * Cm1y.T @ ( dAcorl[:,:,i] @ Cm1y )
            dlogMLcorl[i] = it.item()
         return dlogMLstdM.item(), dlogMLcorl
      else:
         dlogMLcorl = - dataSet.nbValidVectors/2 * np.trace( Am1dAcorl ) + .5 * Cm1y.T @ ( dAcorl @ Cm1y )
         return dlogMLstdM.item(), dlogMLcorl.item() # item because at this point, they're 1x1 arrays

   """
   Gradient method for optimization from a single initialization
   """
   def optimizeGradient( self, dataSet, paramSet0, step=1, H=None ):
      paramSet  = copy(paramSet0)
      paramSetT = copy(paramSet0)
      
      stdMLowerBnd = True # This says that stdM cannot go below a certain value
      if paramSet.stdM == 0.:
         stdMLowerBnd = False

      crit = 1e-6
      
      if paramSet.anisotropy:
         nparams = 1 + dataSet.inputDim
      else:
         nparams = 2
         
#      print(step)
#      print(paramSet.stdM)
#      print(paramSet.corrlen)
      
      for i in range(100): # Gradient method.
         CovM, Cnorm            = self.computeCovarianceNNorm( dataSet, paramSet )
         logML                  = self.computeLogML( dataSet, paramSet, CovM )
         dlogMLstdM, dlogMLcorl = self.computeDLogML( dataSet, paramSet, CovM )
         
         if i>0: # Store previous gradient (for BFGS)
            gradp = grad
         
         if paramSet.anisotropy: # Actually, we minimize -logML, so there is a (-) in the gradient
            grad = -np.concatenate( (np.atleast_1d(dlogMLstdM), dlogMLcorl), axis=0 )
         else:
            grad = -np.array([dlogMLstdM,dlogMLcorl])
         
         ndire2 = np.sum(np.square(grad)) # dlogMLstdM**2 + dlogMLcorl**2 # Norm of the gradient
         if i==0:
            ndire2init = ndire2
         if (ndire2/ndire2init < crit): # Test if convergence criterion was met
            break

         if paramSet.doBFGS and i>0:
            y = grad - gradp
            y = y.reshape((nparams,1))
            dtheta = delta.reshape((nparams,1))
            Hd = H @ dtheta
            H = H + (y @ y.T) / (y.T @ dtheta) - (Hd @ Hd.T) / (dtheta.T @ Hd) # Update Hessian.
            dire = - np.linalg.solve(H,grad) # TODO maybe: try rank one approximation of the inverse
         else:
            dire = -grad # Steepest descent : direction is simply given by the gradient

         # Decreasing linesearch
         didDecrease = False
         for j in range(20): # It should be a while, but i dont want to risk infinite loop
            step = step/2
            delta = step*dire
            
            paramSetT.stdM = paramSet.stdM + delta[0]
            if paramSetT.stdM <= 0: # Std must be > 0
               continue # Need to divide again the step

            if stdMLowerBnd and paramSetT.stdM <= 1e-6*Cnorm: # Ensure regularization is enough for single precision
               continue

            if paramSet.anisotropy:
               paramSetT.corrlenVect = paramSet.corrlenVect + delta[1:]
            else:
               paramSetT.corrlen     = paramSet.corrlen + delta[1]
               paramSetT.corrlenVect = paramSetT.corrlen * np.ones( dataSet.inputDim )

            if np.sum(paramSetT.corrlenVect <= 0): #paramSetT.corrlen <= 0:
               continue # Need to divide again the step
               
#            if np.min(paramSetT.corrlenVect)/np.max(paramSetT.corrlenVect) < paramSetT.minAnisotropyRatio:
#               continue # Stuff is too anisotropic

#            # Final correction to ensure maximal anisotropy
#            if paramSetT.minAnisotropyRatio > 0.:
#               minLen = np.min(paramSetT.corrlenVect)
#               toohigh = (paramSetT.minAnisotropyRatio * paramSetT.corrlenVect > minLen)
#               paramSetT.corrlenVect[toohigh] = minLen/paramSetT.minAnisotropyRatio
            
            CovMT  = self.computeCovariance( dataSet, paramSetT )
            logMLT = self.computeLogML( dataSet, paramSetT, CovMT )

            if (logMLT > logML): # If we are increasing, stop linesearch (its a maximization problem)
               didDecrease = True
               break

         # Prevent exciting this without decreasing.
         if not didDecrease:
            warnings.warn( "Cost function does not want to decrease at iteration " + str(i) )
            break
            
         if i==0 and paramSet.doBFGS and (H is None): 
            H = 1/step * np.identity( nparams ) # Initialize Hessian
            
         paramSet.stdM        = paramSetT.stdM
         paramSet.corrlen     = paramSetT.corrlen
         paramSet.corrlenVect = paramSetT.corrlenVect
         step = 4*step # Increase for next iteration
         
#         print(logML)
#         print(dlogMLstdM)
#         print(dlogMLcorl)
#         print(ndire2)
#         print("=====")
#         print(i)
#         print(step)
#         print(paramSet.stdM)
#         print(paramSet.corrlen)
#         print(paramSet.corrlenVect)
         
         # TODO: convergence condition
         
      #print(logML)
      #print(dlogMLstdM)
      #print(dlogMLcorl)
      #print(paramSet.stdM)
      #print(paramSetT.corrlen)
      #print(i)
      
      # Final correction to ensure maximal anisotropy
      if paramSet.minAnisotropyRatio > 0.:
         minLen = np.min(paramSet.corrlenVect)
         toohigh = (paramSet.minAnisotropyRatio * paramSet.corrlenVect > minLen)
         paramSet.corrlenVect[toohigh] = minLen/paramSet.minAnisotropyRatio
      
      return paramSet, step, H

   """
   Optimizes the parameters via maximization of Log Marginal Likelihood
   """
   def optimizeLogML( self, dataSet ):
      paramSet0 = copy(self.params)  # initializes with given params
      
      if self.params.kernelI is None:
         paramSet1 = copy(paramSet0)
         mystep = 1
         Hessian = None
      else:         # Do a pre-computation with an other kernel
         paramSet0.kernelT = self.params.kernelI
         paramSet1, mystep, Hessian = self.optimizeGradient( dataSet, paramSet0 )
         paramSet1.kernelT = self.params.kernelT # Back to requested kernel
      
      paramSet, _, _ = self.optimizeGradient( dataSet, paramSet1, step=mystep, H=Hessian ) # Calls gradient optimization
      self.params = copy(paramSet)

   """
   Perform Regression
   """
   def regress( self, dataSet, queryPts=None ):
      
      if queryPts is None:
         queryPts = dataSet.queryPts
         nbQuery  = dataSet.nbQuery
      else:
         nbQuery  = np.shape(queryPts)[0]
      
      self.CovM  = self.computeCovariance( dataSet, self.params )                         # Covariance matrix
      self.CoQ   = self.computeCCovariance( dataSet.inputVectors, queryPts, self.params ) # Right Hand Side (= cross covariance)
      
      # Modify Covariance matrix to impose sum(alpha) = 1
      ono = np.ones((dataSet.nbValidVectors,1)) # Row of 1
      on2 = np.ones((nbQuery,1))
      ononm1 = 1/(dataSet.nbValidVectors**2)                # This is the invert of (ono.T @ ono), which is a scalar
      Pbar = ononm1 * (ono @ ono.T)                         # Projector onto image of constraint
      P = np.identity( dataSet.nbValidVectors ) - Pbar      # Projector onto kernel
      a = self.params.std**2 + self.params.stdM**2          # Balance parameter
      
      A = P @ (self.CovM @ P) + a*Pbar  # Modified matrix (LHS for the regression problem)
      b = P @ (self.CoQ - ononm1 * (self.CovM @ (ono @ on2.T))) + ononm1*a * (ono @ on2.T) # Modified Multiple RHS
      
      # Note: this modification prevents A from being sparse. But actually, CovM is not anyways. So this modification is probably better than Lagrange.
      
      """      
      # DEBUG Lagrange way
      A = np.concatenate( (self.CovM,ono), axis=1 )
      ono0 = np.concatenate( (ono.T,np.zeros((1,1))), axis=1 )
      A = np.concatenate( (A,ono0), axis=0)
      b = np.concatenate( (self.CoQ,on2.T), axis=0)
      
      xnmu = np.linalg.solve(A,b)   # DEBUG: Lagrange resolution
      x = xnmu[0:-1,:]
      #print(xnmu)
      #print(x)
      #print(np.shape(xnmu))
      """
      
      x = np.linalg.solve(A,b)  # Solve MRHS system to get coefficients
      self.regsOutput = np.transpose(dataSet.regsVector) @ x # Regression values
      
      # Determine standard deviation : self.params.std**2 - x.T @ b - mu
      mu = ononm1 * ono.T @ ( b - self.CovM @ x ) # Lagrange multiplier (obtained without Lagrange method)
      std2In = self.params.std**2 - mu # Intermediate value
      for i in range(nbQuery):
         xi = x[:,i]
         bi = b[:,i]
         std2In[0,i] = std2In[0,i] - np.dot(xi,bi)

      self.regsOutputStd2 = std2In
      
   """
   Compute derivative of the Regression function
   """
   def Dregress( self, dataSet, queryPts=None ):
      
      if queryPts is None:
         queryPts = dataSet.queryPts
         nbQuery  = dataSet.nbQuery
      else:
         nbQuery  = np.shape(queryPts)[0]
      
      CovM  = self.computeCovariance( dataSet, self.params ) # Data covariance
      DCoQ  = self.computeDCCovariance( dataSet.inputVectors, queryPts, self.params ) # Derivative of Right Hand Side (= cross covariance)
      
      # Modify Covariance matrix to impose sum(alpha) = 1. TODO: function here
      ono = np.ones((dataSet.nbValidVectors,1)) # Row of 1
      on2 = np.ones((nbQuery,1))
      ononm1 = 1/(dataSet.nbValidVectors**2)                # This is the invert of (ono.T @ ono), which is a scalar
      Pbar = ononm1 * (ono @ ono.T)                         # Projector onto image of constraint
      P = np.identity( dataSet.nbValidVectors ) - Pbar      # Projector onto kernel
      a = self.params.std**2 + self.params.stdM**2          # Balance parameter
      
      A = P @ (CovM @ P) + a*Pbar  # Modified matrix (LHS for the regression problem)
      
      Dregs = np.zeros( (nbQuery,dataSet.inputDim) )
      for i in range( dataSet.inputDim ):
         b = P @ (DCoQ[:,:,i] - ononm1 * (CovM @ (ono @ on2.T))) + ononm1*a * (ono @ on2.T) # Modified Multiple RHS
         x = np.linalg.solve(A,b)  # Solve MRHS system to get coefficients
         Dregs[:,i] = np.transpose(dataSet.regsVector) @ x # Regression values

      return Dregs
      
"""
Stand alone utility functions
"""

"""
Computes anisotropically-ponderated squared cross-distance between two multivectors
MyDist2[i,j] = \sum_{n} (u_{in}-u_{jn})**2 / lambd_n**2

Note: function CrossDiff (see above) is not called here because it is less effective in terms of RAM for high dimension
"""
def CrossDist2( u, v, lambd=None ):
   s1 = np.shape(u)[0]
   s2 = np.shape(v)[0]

   dim = np.shape(u)[1]
   if np.shape(v)[1] != dim: # Dimension check
      warnings.warn("multivectors do not have same number of cols!")

   if lambd is None: # No ponderation given: put 1 everywhere
      lambd = np.ones( dim )
      
   MyDist2 = np.zeros( (s1,s2) )        # Initialize
   on1 = np.ones((s1,1))                # Row of 1
   on2 = np.ones((s2,1))
        
   for i in range(dim):
      co1 = np.reshape( u[:,i], (s1,1) )
      co2 = np.reshape( v[:,i], (s2,1) )
      
      left = (co1 @ on2.T)
      righ = (on1 @ co2.T)
      MyDist2 += np.square(left-righ) / lambd[i]**2
      
   return MyDist2
   
   
"""
Compute anisotropically-ponderated difference between two multivectors
MyDiff2[i,j,n] = (u_{in}-u_{jn}) / lambd_n
"""
def CrossDiff( u, v, lambd=None ):
   s1 = np.shape(u)[0]
   s2 = np.shape(v)[0]

   dim = np.shape(u)[1]
   if np.shape(v)[1] != dim: # Dimension check
      warnings.warn("multivectors do not have same number of cols!")

   if lambd is None: # No ponderation given: put 1 everywhere
      lambd = np.ones( dim )
      
   MyDiff = np.zeros( (s1,s2,dim) )        # Initialize
   on1 = np.ones((s1,1))                # Row of 1
   on2 = np.ones((s2,1))
        
   for i in range(dim):
      co1 = np.reshape( u[:,i], (s1,1) )
      co2 = np.reshape( v[:,i], (s2,1) )
      
      left = (co1 @ on2.T)
      righ = (on1 @ co2.T)
      MyDiff[:,:,i] = (left-righ) / lambd[i]
      
   return MyDiff
