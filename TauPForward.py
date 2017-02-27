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
import obspy.taup as tp
from obspy.taup import TauPyModel
import time
from obspy.geodetics.base import kilometer2degrees                                          
                                
NXeq,NYeq,NZeq = 3,3,3
Neq = NXeq*NYeq*NZeq
print ' Setting EQ coordinates...'
xv,yv,zv,stationCoords =  LocationsOnGridSmall(receiver_name = 'receiver.dat',
                                               NX = NXeq, NY=NYeq,NZ=NZeq,leftBottomCorner=[200,200],
                                               rightTopCorner=[5000,5000],depthRange = [4300,4900])
x_perturb = np.random.uniform(low=150,high=700,size =xv.shape)
y_perturb = np.random.uniform(low=150,high=700,size =yv.shape)
z_perturb= np.random.uniform(low=100,high=200,size =zv.shape)
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


name='/home/geoanton/WI_modeling/MCMCTest1.nd'
a = tp.taup_create.build_taup_model(name,output_folder='./ModelObspy')
model= TauPyModel(model='./ModelObspy/MCMCTest1.npz')



t0 = time.time()

N=20

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
            offset = np.sqrt((rowSt.x-eq_coords[0])**2 +(rowSt.y-eq_coords[1])**2)/1000.0
            arrivals = model.get_travel_times(source_depth_in_km=eq_coords[2]/1000.0,
                                              distance_in_degree=kilometer2degrees(offset),
                                  receiver_depth_in_km=0.01,phase_list=['p'])
            p=arrivals[0].time
            tp[eq_index,st_index]=p 
            
            #print ' Done with station %d and eq %d ' % (st_index,eq_index)
    print ' Done with iter %d  ' % (ii)


t1 = time.time()
total = t1-t0  
print 'Total time is %3.6f s' % total      












