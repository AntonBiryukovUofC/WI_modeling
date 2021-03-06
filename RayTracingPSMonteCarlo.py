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
from MiscFunctions import GetPSArrivalRayTracingMC,getNonLinLocPhaseLine
import obspy

import time


# Create a model for ray-tracing:
Log = pd.read_csv('LogRepsol.csv')
SourcesCSV = pd.read_csv('sourcesDF.csv') # Read the source locations

Log =Log[Log.Depth <6000]
z_layers = np.linspace(900,5000,30)/ 1000.
z_layers[0] = 0

Log['VpMean'] = pd.Series(np.ones(Log.shape[0]))
Log['VsMean'] = pd.Series(np.ones(Log.shape[0]))


Log.Depth=Log.Depth/1000
Log.Vp=Log.Vp/1000
Log.Vs=Log.Vs/1000
NDFile = ''
discontinuities=[]
   
for i in range(len(z_layers)-1):
    mask =(Log.Depth>=z_layers[i]) & (Log.Depth<=z_layers[i+1])
    if not(mask.any()):
        break
    Log.loc[mask,'VpMean'] = Log.loc[mask,'Vp'].mean()
    Log.loc[mask,'VsMean'] = Log.loc[mask,'Vs'].mean()
    Log.loc[mask,'RhoMean'] = Log.loc[mask,'Rho'].mean()
    #Layer = Layer(ztop = z_layers[i],zbot = z_layers[i+1])
    layer_strTop = "%.3f %.3f %.3f %.3f\n" % (
                                           z_layers[i],Log.loc[mask,'Vp'].mean(),
                                           Log.loc[mask,'Vs'].mean(),
                                           Log.loc[mask,'Rho'].mean()
                                           )
    layer_strBot = "%.3f %.3f %.3f %.3f\n" % (
                                           z_layers[i+1],Log.loc[mask,'Vp'].mean(),
                                           Log.loc[mask,'Vs'].mean(),
                                           Log.loc[mask,'Rho'].mean()
                                           )
    nameD = 'nd%d\n' % i  
    discontinuities.append('nd%d' % i)                                          
    NDFile = NDFile  + layer_strTop + layer_strBot + nameD
Log.loc[Log.shape[0]-1,'VpMean']=Log.loc[Log.shape[0]-2,'VpMean']


layer_strBot = "%.3f %.3f %.3f %.3f\n" % (
                                           z_layers[i+1]*1.05,Log.loc[mask,'Vp'].mean(),
                                           Log.loc[mask,'Vs'].mean(),
                                           Log.loc[mask,'Rho'].mean()
                                           )

NDFile = NDFile + layer_strBot
#################################################################################################################
# Load the model
with open(('RepsolHighRes.nd'),'w') as f:
    f.write(NDFile)

model =cake.load_model(('VpVs.nd'))



#model =cake.load_model('VpVs.nd')
MC = 100000
nLayers = 6
pertVp = np.zeros((MC,nLayers))
pertVs = np.zeros((MC,nLayers))
tops= np.zeros(nLayers)
bots = np.zeros(nLayers)
k=0.00001
i=0
MCDf=pd.DataFrame()
for l in model.layers():
    
    print l.mtop.vp/l.mtop.vs
    pertVp[:,i] = k*l.mtop.vp * np.random.randn(MC) + l.mtop.vp
    pertVs[:,i] = k*l.mtop.vs * np.random.randn(MC) + l.mtop.vs
    tops[i] = l.ztop
    bots[i]=l.zbot
    i+=1
    
    
       
stationsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekStations.csv')
stCoords = np.zeros((stationsCatalog.shape[0],3))
Xref = stationsCatalog.X[3]
Yref = stationsCatalog.Y[3]
stCoords[:,0]=stationsCatalog.X - Xref
stCoords[:,1]=stationsCatalog.Y - Yref
stCoords[:,2]=10

kk=0
# Check the ratio for VPVS:
for i in range(MC):
    VpVs = pertVp[i,:]/pertVs[i,:]
    if np.any(VpVs < 1.5):
        ind  = np.where(VpVs <1.6)
        pertVs[i,ind]=pertVp[i,ind]/1.6
        print "changed the bad ratio !"
        kk=kk+1

p_arrival = np.zeros((MC,stationsCatalog.shape[0]))
s_arrival = np.zeros((MC,stationsCatalog.shape[0]))
so_offset = np.zeros((MC,stationsCatalog.shape[0]))
# Time the calculations here !

t0 = time.time()
model =cake.load_model(('VpVs.nd'))

for iMC in range(MC):
    i=0
    # Perturb the model:
    for l in model.layers():
        l.mtop.vp = pertVp[iMC,i]
        l.mtop.vs = pertVs[iMC,i]
        i+=1    
    
 
    
    for iSt in range(stCoords.shape[0]):
        p_arrival[iMC,iSt],s_arrival[iMC,iSt],so_offset[iMC,iSt] = GetPSArrivalRayTracingMC(
                                                            sta_coords=stCoords[iSt,:],
                                                            eq_coords=np.array([4000,6000,3910]),
                                                            model=model)
        print ' Done with station %d and MCiter %d ' % (iSt,iMC)
        
t1 = time.time()
total = t1-t0  
print 'Total time is %3.6f' % total      
        
'''
for iSt in range(4):
    MCDf['Psta%d' % (iSt+1)] = p_arrival[:,iSt]
    MCDf['Ssta%d' % (iSt+1)] = s_arrival[:,iSt]
    MCDf['Dsta%d' % (iSt+1)] = so_offset[:,iSt]

MCDf.to_csv('MonteCarloPicks.csv')
MCDf.ID  = range(MC)
MCDf.index = MCDf.ID

de=obspy.UTCDateTime('1992-01-01')
   
# Create observation files for NONLINLOC
if not(os.path.exists('./MCPhases')):
    os.makedirs('./MCPhases')
    
MCDf.dropna()
for i,row in MCDf.dropna().iterrows():
    de.minute=np.random.randint(low=0,high=60)
    de.hour=np.random.randint(low=0,high=23)
    de.julday=np.random.randint(low=1,high=364)
    eventObs=''
    iSt=0
    for ii,rowSt in stationsCatalog.iterrows():
        phase_completeP = getNonLinLocPhaseLine(de = de+row['Psta%d' % (iSt+1)],sta=rowSt.Name,ch='Z',phase = 'P')
        phase_completeS = getNonLinLocPhaseLine(de = de+row['Ssta%d' % (iSt+1)],sta=rowSt.Name,ch='Z',phase = 'S')
        iSt+=1
        eventObs+=phase_completeP+phase_completeS
        print phase_completeP
    with open('./MCPhases/event%03d.obs' % i,'w') as f:
        f.write(eventObs)











pertVp1 = pertVp[MCDf.dropna().index,:] 
pertVs1 = pertVs[MCDf.dropna().index,:] 


fig,ax = plt.subplots(ncols=2,figsize=(4,6))

norm1=mpl.colors.Normalize(vmin=2300,vmax=6000)
m = cm.ScalarMappable(norm=norm1, cmap=cm.jet)


for j in range(nLayers): # plot histograms of distribution!
    # Now for Vp
    values,bins = np.histogram(pertVp1[:,j],bins=50,density=True)
    bins = (bins[0:-1]+bins[1:])/2.0
    values = values/values.max()
    values = np.vstack((values,values))
    x,y = np.meshgrid(bins,[tops[j],bots[j]])
    pc = ax[0].pcolormesh(x,y,values,vmin=0,vmax=values.max(),cmap=cm.jet,shading='gouraud')
    # Now for Vs
    values,bins = np.histogram(pertVs1[:,j],bins=50,density=True)
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
ax[0].set_xlim((2000,7500))
ax[0].set_ylabel('Depth [m]')
ax[1].set_xlim((1000,5000))
ax[0].set_xticks([2000,4000,6000,8000])
ax[1].set_xticks(range(1000,6000,1000))

fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
plt.colorbar(mappable = pc,cax=cax)
ax[0].set_title('Vp [m/s]')
ax[1].set_title('Vs [m/s]')
ax[1].set_yticklabels([])
cax.set_ylabel('PDF, normalized by max.')
fig.savefig('VpVs_after.png',dpi=200,bbox_inches='tight')
# Set up here a catalog in Obspy fashion
'''



