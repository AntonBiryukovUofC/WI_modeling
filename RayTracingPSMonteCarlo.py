# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
mpl.style.use('ggplot')
import os
from LocationsOnGrid import LocationsOnGridSmall
from MiscFunctions import GetPSArrivalRayTracingMC


model =cake.load_model('VpVs.nd')
SourcesCSV = pd.read_csv('sourcesDF.csv') # Read the source locations
MC = 2000
nLayers = 6
pertVp = np.zeros((MC,nLayers))
pertVs = np.zeros((MC,nLayers))
tops= np.zeros(nLayers)
bots = np.zeros(nLayers)
k=0.05
i=0
MCDf=pd.DataFrame()
fig,ax = plt.subplots(ncols=2,figsize=(8,10))

norm1=mpl.colors.Normalize(vmin=2300,vmax=6000)
m = cm.ScalarMappable(norm=norm1, cmap=cm.jet)
for l in model.layers():
    
    print l.mtop.vp/l.mtop.vs
    pertVp[:,i] = k*l.mtop.vp * np.random.randn(MC) + l.mtop.vp
    pertVs[:,i] = k*l.mtop.vs * np.random.randn(MC) + l.mtop.vs
    tops[i] = l.ztop
    bots[i]=l.zbot
    i+=1
    
for j in range(nLayers): # plot histograms of distribution!
    # Now for Vp
    values,bins = np.histogram(pertVp[:,j],bins=50,density=True)
    bins = (bins[0:-1]+bins[1:])/2.0
    values = values/values.max()
    values = np.vstack((values,values))
    x,y = np.meshgrid(bins,[tops[j],bots[j]])
    pc = ax[0].pcolormesh(x,y,values,vmin=0,vmax=values.max(),cmap=cm.jet,shading='gouraud')
    # Now for Vs
    values,bins = np.histogram(pertVs[:,j],bins=50,density=True)
    values = values/values.max()
    
    bins = (bins[0:-1]+bins[1:])/2.0
    values = np.vstack((values,values))
    x,y = np.meshgrid(bins,[tops[j],bots[j]])
    ax[1].pcolormesh(x,y,values,vmin=0,vmax=values.max(),cmap=cm.jet,shading='gouraud')

    
for axInd in ax.flatten():
    axInd.set_ylim((0,7000))
    axInd.invert_yaxis()
    for tick in axInd.get_xticklabels():
        tick.set_rotation(45)
ax[0].set_xlim((2000,7000))
ax[1].set_xlim((1000,5000))
plt.colorbar(mappable = pc)
ax[0].set_title('Vp')
ax[1].set_title('Vs')


    
    
    

_,_,_,stCoords = LocationsOnGridSmall(receiver_name='receiver.dat',NX=1,NY = 1,NZ =1) # Get the receiver locations
kk=0
# Check the ratio for VPVS:
for i in range(MC):
    VpVs = pertVp[i,:]/pertVs[i,:]
    if np.any(VpVs < 1.5):
        ind  = np.where(VpVs <1.5)
        pertVs[i,ind]=pertVp[i,ind]/1.6
        print "changed the bad ratio !"
        kk=kk+1

iSt=0
p_arrival = np.zeros(MC)
s_arrival = np.zeros(MC)
so_offset = np.zeros(MC)
for iMC in range(MC):
    model =cake.load_model('VpVs.nd')
    i=0
    # Perturb the model:
    for l in model.layers():
        l.mtop.vp = pertVp[iMC,i]
        l.mtop.vs = pertVs[iMC,i]
        i+=1    
    
 
    
    
    p_arrival[iMC],s_arrival[iMC],so_offset[iMC],model = GetPSArrivalRayTracingMC(
                                                            sta_coords=stCoords[iSt,:],
                                                            eq_coords=np.array([2500,2500,3910]),
                                                            model=model)
    print ' Done with station %d and MCiter %d ' % (iSt,iMC)
    
MCDf['Psta%d' % (iSt+1)] = p_arrival
MCDf['Ssta%d' % (iSt+1)] = s_arrival
MCDf['Dsta%d' % (iSt+1)] = so_offset

MCDf.to_csv('MonteCarloPicks.csv')


