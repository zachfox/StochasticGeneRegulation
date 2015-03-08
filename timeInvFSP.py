# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 14:22:14 2014

@author: Zach Fox
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as mpl  

class easyFSP:
    '''
    This is a class to compute the FSP for a time invariant A matrix.
    '''
    
    def __init__(self,numSpecies, A_Matrix,errorTolerance,initialState,finalTime,initialTime):
        ################################################################
        #The state transition matrix (A matrix) is expected to be a 2D
        #numpy array. As of now, error tolerance yield binary response, and
        #the A matrix must be resized manually. 
        ################################################################
        self.numSpecies = numSpecies
        self.A = A_Matrix
        self.tol = errorTolerance
        self.x0 = initialState 
        self.tf = finalTime
        self.ti = initialTime
        
        
    def FSPCalc(self):
        ################################################################            
        #Computes the Finite State Projection for time independent systems
        #within given error.
        #A is the state reaction matrix, tf is the final time of the projections, 
        #X_0 is the initial states, error is the amount of error between the FSP and 
        #the actual solution.    
        ################################################################
        
        ones = np.array([(np.ones(np.shape(self.A[1])))])   #computes a 2D row of 1's.
        matexp = linalg.expm(self.A*self.tf)                #compute matrix exponential
        G = np.dot((np.dot(ones,matexp)),self.x0) 
        if G >= 1-self.tol:                 
            xf= np.dot(matexp,self.x0)       
            return xf
        else: 
            print('FSP is a no go, sire')        
            #states+=1                                  #increase number of states
            #FSPAlgorithm(A,tf,X_0,error,states)        #recursive step
    
    