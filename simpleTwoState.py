# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:40:49 2015

@author: Zach Fox
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
kon,koff,kr1,deg = [2.0,1.5,10.0,.30]
propensityVec = [(kon,0),(koff,1),(kr1,1),(deg,2)]
stoichiometryMatrix = np.concatenate((r1.T,r2.T,r3.T,r4.T),axis=1)
#pick some times
tstart = 0
tend = 50
numTimes = 30
tspace = np.linspace(tstart,tend,numTimes)

#run a single trajectory
B = SSA.generalSSA(stoichiometryMatrix,x0,propensityVec,tstart,tend,numTimes)
D = B.runTrajectoryDirect()

#Plotting using built-in plotting
B.plotTrajectories([2,1,0],['mRNA','on state','off state'])
#Make an animation
B.makeTrajectoryAnimation([2,1,0],['mRNA','on state','off state'])

#make a distribution 
numTrajectories = 150
G = B.makeDistribution(numTrajectories)
plt.figure()
plt.hist(G[-1,-2,:],bins=20,color = 'r')

                  