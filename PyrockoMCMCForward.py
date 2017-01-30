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
                                                            
                                
NXeq,NYeq,NZeq = 5,5,3
Neq = NXeq*NYeq*NZeq
xv,yv,zv,stationCoords =  LocationsOnGridSmall(receiver_name = 'receiver.dat',
                                               NX = NXeq, NY=NYeq,NZ=NZeq,leftBottomCorner=[1000,1000],
                                               rightTopCorner=[5000,5000],depthRange = [4700,6000])
x_perturb = np.random.uniform(low=50,high=400,size =xv.shape)
y_perturb = np.random.uniform(low=50,high=400,size =yv.shape)
z_perturb= np.random.uniform(low=10,high=200,size =zv.shape)
xv +=x_perturb
yv+=y_perturb
zv+=z_perturb
#returnn
# stCoords - create on random
print 'Setting the station coordinates...'
Nstations=5
np.random.seed(seed=1)
x = np.random.uniform(low=200,high = 7000,size =Nstations)
y = np.random.uniform(low=200,high = 7000,size =Nstations)
z = 10*np.ones_like(x)
stCoords = np.stack((x,y,z),axis=1)
# Throw the coordinates of the stations and the earthquakes into corresponding DataFrames
stdf=pd.DataFrame({'x':x,'y':y,'z':z})
eqdf=pd.DataFrame({'x':xv.flatten(),'y':yv.flatten(),'z':zv.flatten()})

fig, ax = plt.subplots()
sns.regplot(x='x',y='y',data = stdf,fit_reg=False,scatter_kws = {'s':60},ax=ax)
sns.regplot(x='x',y='y',data = eqdf,fit_reg=False,scatter_kws = {'s':30},ax=ax,color='r')

# Number of iterations
N=20
# Time the calculations here !

t0 = time.time()
model =cake.load_model(('MCMCTest.nd'))

for ii in range(N):
    tp=np.zeros((Neq,Nstations))
    ts=np.zeros_like(tp)
    so=np.zeros_like(tp)

    #inputs = [input_list for x in range(MC)]
    #num_cores =multiprocessing.cpu_count()-2
    #results = Parallel(n_jobs=num_cores)(delayed(GetRayTracingPar)(i) for i in inputs)        
    for eq_index,rowEq in eqdf.iterrows():
        eq_coords = np.array([rowEq.x,rowEq.y,rowEq.z])
        for st_index,rowSt in stdf.iterrows():
            # Get the time for P,S arrivals and offset
            p,s,o= GetPSArrivalRayTracingMC(sta_coords=[rowSt.x,rowSt.y,rowSt.z],
                                               eq_coords=eq_coords,
                                               model=model)
            tp[eq_index,st_index],ts[eq_index,st_index],so[eq_index,st_index]=p,s,o 
            
            #print ' Done with station %d and eq %d ' % (st_index,eq_index)
    print ' Done with iter %d  ' % (ii)
             

        
        
t1 = time.time()
total = t1-t0  
print 'Total time is %3.6f s' % total      

np.savez('ForwardDataMCMC.npz',tp=tp,ts=ts,so=so,stdf=stdf,eqdf=eqdf)        
 