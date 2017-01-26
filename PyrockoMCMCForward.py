# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import GetPSArrivalRayTracingMC
from joblib import Parallel, delayed
import multiprocessing
from LocationsOnGrid import LocationsOnGridSmall
# For plotting / data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import time
# Load the model
def GetRayTracingPar(input_list):
    p_arrival,s_arrival,so_offset = GetPSArrivalRayTracingMC(
                                                            sta_coords=input_list[0],
                                                            eq_coords=input_list[1],
                                                            model=input_list[2])
    return p_arrival,s_arrival,so_offset
                                                            
                                

xv,yv,zv,stationCoords =  LocationsOnGridSmall(receiver_name = 'receiver.dat',
                                               NX = 5, NY=5,NZ=3,leftBottomCorner=[1000,1000],
                                               rightTopCorner=[5000,5000],depthRange = [4700,6000])
#returnn
# stCoords - create on random
print 'Setting the station coordinates...'
Nstations=5
np.random.seed(seed=1)
x = np.random.uniform(low=200,high = 7000,size =Nstations)
y = np.random.uniform(low=200,high = 7000,size =Nstations)
z = 10*np.ones_like(x)
stCoords = np.stack((x,y,z),axis=1)
stdf=pd.DataFrame({'x':x,'y':y,'z':z})
sns.lmplot(x='x',y='y',data = stdf,fit_reg=False,scatter_kws = {'s':60})

# Number of iterations
MC = 10000

# Time the calculations here !

t0 = time.time()
model =cake.load_model(('MCMCTest.nd'))

input_list = [stCoords[0,:],np.array([4000,6000,3910]),model]
    
inputs = [input_list for x in range(MC)]
num_cores =multiprocessing.cpu_count()-2
results = Parallel(n_jobs=num_cores)(delayed(GetRayTracingPar)(i) for i in inputs)        
        
        
        
        
t1 = time.time()
total = t1-t0  
print 'Total time is %3.6f s' % total      
 