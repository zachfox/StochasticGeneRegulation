# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 15:40:40 2015

@author: root
"""
import matplotlib.pyplot as mpl
import numpy as np
import timeInvFSP as FSP

'''
Example from JCP publication in which the exact solution of the CME
can be calculated by the FSP. See this link for the system 
http://cbe.colostate.edu/~munsky/Papers/JChemPhys_124_044104.pdf   
'''    

#define parameters    
r0=5.; r=r0;
u0=100.;  
c1=1.; c2=2.5-(2.25*(r/(1.+r))); c3=1.;
c4=1.2-(.2*(r/(1+r))); c5=.01;
c6=c4; c7=.01; c8=c2;

#Define state transition matrix 
A=np.array([[-c1*u0-c3*u0, c2, c4, 0],
            [c1*u0,-c2-c5*(u0-1), 0, c6],
            [c3*u0, 0, -c4-c7*(u0-1), c8],
            [0, c5*(u0-1), c7*(u0-1), -c6-c8]])

#define initial condition for each gene state           
X_0=np.array([[1,0,0,0]]).T
numSpecies = len(X_0)
error=10e-6
t0=0; tf=10;

#Initialize class & compute FSP 
B = FSP.easyFSP(numSpecies,A,error,X_0,tf,t0)
soln = B.FSPCalc()

#print check the sum of probabilities 
print(np.sum(soln.T))

#plotting
x=range(4)
mpl.bar(x,soln,width=.8,align='center')
mpl.xticks(x,['g1', 'g2', 'g3', 'g4'])
mpl.show()
