import numpy as np
import scipy.optimize as opt
from LocationsOnGrid import LocationsOnGridSmall
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
from MiscFunctions import GetHjVjRhoj,costFunc,CalculatePTime,DoForwardModel_MyTracer



np.random.seed(seed=1)
                        
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

mname = 'MCMCTest'
vels = np.array([   3100,   4470,   6200])
rhos = np.array([2.32,2.55,2.75,2.32,2.55,2.75])
depths = np.array([ 2000,   4000])
N=1

t0 = time.time()

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
            offset = np.sqrt((rowSt.x-eq_coords[0])**2 +(rowSt.y-eq_coords[1])**2)
            source_depth=eq_coords[2]
            #p,r= CalculatePTime(vels=vels,depths=depths,rhos=rhos,
             #                 source_offset=offset,
              #                source_depth=source_depth)
            p,r= CalculatePTime(vels=vels,depths=depths,rhos=rhos,
                              source_offset=offset,
                              source_depth=source_depth,costFunc=costFunc)
            tp[eq_index,st_index]=p
            so[eq_index,st_index]=offset
            
   
            #print ' Done with station %d and eq %d ' % (st_index,eq_index)
    print ' Done with iter %d  ' % (ii)
             

        
        
t1 = time.time()
total = t1-t0  




print 'Total time is %3.6f s' % total      




np.savez('Forward%s.npz' % mname,tp=tp,ts=ts,so=so,stdf=stdf,eqdf=eqdf)        

'''
0.000 3.100 1.500 2.320
2.000 3.100 1.500 2.320
nd0
2.000 4.470 1.500 2.550
4.000 4.470 1.500 2.550
nd4
4.000 6.200 1.500 2.750
7.000 6.200 1.500 2.75
'''


