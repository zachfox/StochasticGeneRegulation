# -*- coding: utf-8 -*-
"""
@author: ZachFox
"""

import numpy as np

class generalSSA:
    
    def __init__(self,stoichiometryMatrix,initialConditions,propensityVec,t0,tend,numTimes,fastRxn=False):
        
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
        #[(ki,xj),function(x,t),...]
        #######################################################################
        
        self.propensityCoefficients = propensityVec
        
        #define times to record
        self.t0 = t0 
        self.tend = tend
        self.numTimes = numTimes
        self.tspace = np.linspace(t0,tend,numTimes)
        #define "fast" reaction for timekeeping
        if fastRxn is True:
            self.fastrxn = .5
        else: 
            self.fastrxn = .0000000000000001
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
        distributionArray = np.zeros((self.numSpecies,self.numTimes,numTrajectories))
        for i in range(numTrajectories):
            distributionArray[:,:,i] = self.runTrajectoryDirect()
        return distributionArray
    
    def makeDistributionMPI(self,numTrajectories):
        ###############################################
        #Parallelized SSA runs are implemented using mpi4py
        #distribution array is 3D, organized as follows 
        #[species,times,trajectoryNumber]
        ###############################################
        from mpi4py import MPI
        self.distributionArray = np.zeros((self.numSpecies,self.numTimes,numTrajectories))
        #Initialize MPI
        comm = MPI.COMM_WORLD
        rank,size = comm.Get_rank(), comm.Get_size()
        print size
        if rank == 0:
            bigLoop = np.arange(numTrajectories)
            smallLoops = np.array_split(bigLoop,size)
        else:
            smallLoops = None
        smallLoop = comm.scatter(smallLoops,root=0)
        smallLoop = self.makeDistribution(len(smallLoop))
    
        gathered_chunks = comm.gather(smallLoop,root=0)
        if rank ==0:
            #print np.shape(gathered_chunks)
            grouping = np.concatenate(gathered_chunks,axis=2)
            print np.shape(grouping)
            return grouping
        else:
            gathered_chunks=None
            grouping = None
    

    def plotTrajectories(self,speciesID,speciesNames = None):
        ######################################################
        #This is a function to easily plot a single ssa trajectory
        #For each species that the user wants to see. This is for convenience:
        #If a more complex plot is desired the user can still easily accomplish
        #this manually. 
        ######################################################
        import matplotlib.pyplot as plt
        numSpecies = len(speciesID)
        f,axarr = plt.subplots(nrows=numSpecies,ncols=1)
        for i in range(numSpecies):
            axarr[i].step(self.tspace,self.solutionVec[speciesID[i],:]) 
            axarr[i].set_ylim([0, 1.2*np.max(self.solutionVec[speciesID[i],:])])
            if speciesNames is not None:
                axarr[i].set_ylabel(speciesNames[i])
        axarr[-1].set_xlabel('Time')
        f.show()
        return (f,axarr)
        
    def makeTrajectoryAnimation(self,speciesID,speciesNames = None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        #Set up subplots
        l=[]
        numSpecies = len(speciesID)
        f,axarr = plt.subplots(nrows=numSpecies,ncols=1)
        for i in range(numSpecies):
            l.append(axarr[i].step([],[]))
            axarr[i].set_ylim([0, 1.2*np.max(self.solutionVec[speciesID[i],:])])
            if speciesNames is not None:
                axarr[i].set_ylabel(speciesNames[i])
        axarr[-1].set_xlabel('Time')
        #Get solutions from solutionVec
        data = self.solutionVec[speciesID,:]
        
        def update_lines(num, data, axarr):
            m=0
            for i in axarr:
                stuff =  np.vstack((self.tspace,data[m,:]))
                print num
                print stuff[...,:num]
                i[0].set_data(stuff[...,:num])
                m+=1
            return axarr[0][0],axarr[1][0],axarr[2][0]           
            
        line_ani = animation.FuncAnimation(f, update_lines, self.tspace, fargs=(data, l),
        interval=50, blit=True)
        plt.show()
        
        
        
        
        

        
        
        
        