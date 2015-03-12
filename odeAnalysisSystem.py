# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:01:50 2015

@author: Zach Fox
"""

import numpy as np
import scipy.integrate as ode 

class odeAnalysis:
    
    def __init__(self,initialConditions,propensityVec,t0,tend,numTimes):

        #define initial conditions
        self.x0=initialConditions
        
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


    def makeODEMatrix(self,t,y):
        #############################################
        #make a matrix in the form dX/dt = A X
        #############################################
        x0=np.array([-k12, -k21-k23, -k32-k34,-k43,0])
        x1=np.array([k21,k32,k43,0])
        x_1=np.array([k12,k23,k34,0])
        r4=np.array([kr1,kr2,kr3,kr4,-d])
        mat=np.diag(x0,k=0)+np.diag(x1,k=1)+np.diag(x_1,k=-1)
        mat[4:]=r4
        return mat
    
def deriv(t,y,params,hogParams):
    mat=odeMatrix(t,y,params,hogParams)
    eqns=np.dot(mat,y)
    return eqns

def main(tend,parameter,numPoints):
    #start_time=time.time()
    t_final=tend
    t0=0
    dt=10
    params =import_sysparams(0)
    hogs=import_hogparams(0)
    params[3]=parameter[0]
    params[4]=parameter[1]
    num_steps = np.floor((t_final - t0)/dt) + 1
    y0=np.ravel(np.array([[1.0,0.0,0.0,0.0,0.0]]).T)
    ode15s = integrate.ode(deriv,odeMatrix)
    ode15s.set_integrator('lsoda',with_jacobian=True)
    ode15s.set_initial_value(y0,t0)
    ode15s.set_f_params(params,hogs)
    ode15s.set_jac_params(params,hogs)
    t=np.zeros((num_steps,1))
    y=np.zeros((num_steps,np.size(y0)))
    t[0]=0
    y[0,:]=y0
    k = 1
    while ode15s.successful() and k < num_steps:
        ode15s.integrate(ode15s.t+dt)
        t[k] = ode15s.t
        y[k,:] = ode15s.y
        k+=1
#    plt.figure(3)
#    plt.plot(t/60.,y[:,4])
#    plt.xlabel('Minutes')
#    plt.ylabel('mRNA Level')
#    plt.ylim((0,50))
#    plt.show()
    
    #print "ODE Simulation Time: %f seconds" % (time.time()-start_time) 
    odeSoln=y[:,4]
    alpha=float(tend)/(dt*numPoints)
    sampledSoln=np.zeros(numPoints+1)
    tNew=np.zeros(numPoints+1)
    for i in range(numPoints+1):
        sampledSoln[i]=odeSoln[alpha*i]
        tNew[i]=t[alpha*i]
    #plt.plot(tNew,sampledSoln)
    return sampledSoln