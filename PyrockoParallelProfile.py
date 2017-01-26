# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import GetPSArrivalRayTracingMC
from joblib import Parallel, delayed
import multiprocessing


import time
# Load the model
def GetRayTracingPar(input_list):
    p_arrival,s_arrival,so_offset = GetPSArrivalRayTracingMC(
                                                            sta_coords=input_list[0],
                                                            eq_coords=input_list[1],
                                                            model=input_list[2])
    return p_arrival,s_arrival,so_offset
                                                            
                                

model =cake.load_model(('VpVs.nd'))


# Number of iterations
MC = 10000
nLayers = 6
       
stationsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekStations.csv')
stCoords = np.zeros((stationsCatalog.shape[0],3))
Xref = stationsCatalog.X[3]
Yref = stationsCatalog.Y[3]
stCoords[:,0]=stationsCatalog.X - Xref
stCoords[:,1]=stationsCatalog.Y - Yref
stCoords[:,2]=10


p_arrival = np.zeros((MC,stationsCatalog.shape[0]))
s_arrival = np.zeros((MC,stationsCatalog.shape[0]))
so_offset = np.zeros((MC,stationsCatalog.shape[0]))
# Time the calculations here !

t0 = time.time()
model =cake.load_model(('VpVs.nd'))
input_list = [stCoords[0,:],np.array([4000,6000,3910]),model]
    
inputs = [input_list for x in range(MC)]
num_cores =multiprocessing.cpu_count()-2
results = Parallel(n_jobs=num_cores)(delayed(GetRayTracingPar)(i) for i in inputs)        
        
        
        
        
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



