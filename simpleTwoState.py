# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:40:49 2015

@author: root
"""
import numpy as np
import stochasticSimulationAlgorithm as SSA
import matplotlib.pyplot as plt

"""
This code is a simple two gene-state model, with one state corresponding to 
no transcription, another corresponding to transcription.
"""
#off to on
r1 = np.array([[-1.0,1.0,0]])
#on to off
r2 = np.array([[1.0,-1.0,0]])
#transcription
r3 = np.array([[0,0,1.0]])
#degradation
r4 = np.array([[0,0,-1.0]])
#initial conditions
x0 = [0,1,0]
#propensity vector
kon,koff,kr1,deg = [1.0,.1,30.0,.30]
propensityVec = [(kon,0),(koff,1),(kr1,1),(deg,2)]
stoichiometryMatrix = np.concatenate((r1.T,r2.T,r3.T,r4.T),axis=1)

#run a single trajectory
B = SSA.generalSSA(stoichiometryMatrix,x0,propensityVec,0,200,50)
D = B.runTrajectoryDirect()

#plotting
f,axarr = plt.subplots(nrows=3,ncols=1)
axarr[0].plot(np.linspace(0,200,50),D[2,:])
axarr[1].plot(np.linspace(0,200,50),D[1,:])
axarr[2].plot(np.linspace(0,200,50),D[0,:])
plt.xlabel('time')
axarr[0].set_ylabel('number of mRNA')
axarr[1].set_ylabel('on state')
axarr[2].set_ylabel('off state')
axarr[1].set_ylim([0,1.3])
axarr[2].set_ylim([0,1.3])

                      