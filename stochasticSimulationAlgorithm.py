# -*- coding: utf-8 -*-
"""
@author: ZachFox
"""

import numpy as np

class generalSSA:
    
    def __init__(self,stoichiometryMatrix,initialConditions,propensityVec,t0,tend,numTimes):
        
        #define initial conditions
        if len(initialConditions)!=len(stoichiometryMatrix[:,0]):
            print 'wrong number of initial conditions'
            return
        self.x0=initialConditions
        
        #define number of species
        self.numSpecies=np.size(self.x0)
        
        #define stoichiometry matrix
        A,B=np.shape(stoichiometryMatrix)
        Y=np.zeros((A,B+1))
        Y[:,1:] = stoichiometryMatrix
        self.stoich_mat=Y
        
        #######################################################################
        #define propensity functions/coefficients
        #The propensity vector should be in order of REACTIONS
        #and of the form [(ki,xj)...],where ki and xj are the propensity coef
        #of reaction i and the species x upon which the reactions is proportional. 
        #xj can be a list of indeces or a single int. If propensity functions
        #are something different (hill,M-M,some time dependent function), 
        #make the element in propensityVec that particular function, f(x,t). 
        #######################################################################
        
        self.propensityCoefficients = propensityVec
        
        #define times to record
        self.t0 = t0 
        self.tend = tend
        self.numTimes = numTimes
        
        #define "fast" reaction for timekeeping
        self.fastrxn = .5
    
    def propensityFunction(self,x,t):
        x=np.array(x)
        propensity = np.zeros(len(self.propensityCoefficients)+1)
        propensity[0] = self.fastrxn
        index = 1
        for i in self.propensityCoefficients:
            if type(i) == tuple: 
                propensity[index]=i[0]*np.prod(x[i[1]])
            else:
                propensity[index]=i(x,t)
            index+=1
        return propensity
                
        
    def runTrajectoryDirect(self):
        tspace=np.linspace(self.t0,self.tend,self.numTimes)
        tindex=1
        t = self.t0
        
        self.solutionVec=np.zeros((self.numSpecies,np.size(tspace)))
        self.solutionVec[:,0]=np.ravel(self.x0)
        x=np.array(self.x0)
        while t<self.tend:
            a=self.propensityFunction(x.T,t)
            a0=np.sum(a)
            tau = np.random.exponential(1.0/a0)
            t+=tau #time to next reaction
            if t<=self.tend:
                while t>tspace[tindex]:
                    self.solutionVec[:,tindex]=np.ravel(x)
                    tindex+=1    
                r2 = np.random.uniform(0,1)*a0;
                i=1
                while np.sum(a[0:i])<r2:
                    i+=1
                i-=1
                x=(x.T+self.stoich_mat[:,i]).T  
        b=np.size(self.solutionVec[1,tindex:]) 
        self.solutionVec[:,tindex:]=np.tile(np.array([self.solutionVec[:,tindex-1]]).T,b)
        return self.solutionVec
    
    def makeDistribution(self, numTrajectories):
        ###############################################
        #distribution array is 3D, organized as follows 
        #[species,times,trajectoryNumber]
        ###############################################
        self.distributionArray = np.zeros((self.numSpecies,self.numTimes,numTrajectories))
        for i in range(numTrajectories):
            self.distributionArray[:,:,i] = self.runTrajectoryDirect()
        return self.distributionArray
        
        