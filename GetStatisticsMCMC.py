# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.externals import joblib        
import pylab as pl
from scipy.interpolate import interp1d
from MiscFunctions import DoForwardModel,GetPSArrivalRayTracingMC,MakeModel
# For plotting / data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,uniform,norm
import time
import matplotlib.collections as mc
        

#######################################################################
np.random.seed(1) # set the seed
# Load the data which will be fitted 
data = np.load('ForwardMCMCTest.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']

eqdf = pd.DataFrame(data=eqdf,columns=['x','y','z'])
stdf = pd.DataFrame(data=stdf,columns=['x','y','z'])

model_file = np.load('models_PCA.npz')
burn_in = 100

models=model_file['models']
FinalIter = model_file['iter']
LL=model_file['LL'][burn_in:FinalIter]
tp_array=model_file['tp_array'][burn_in:FinalIter]
log_likelihood_true = model_file['LL_true']

NMod=models.shape[0]
ModelsDF=pd.DataFrame({'Vp1':models[burn_in:FinalIter,0],
                       'Vp2':models[burn_in:FinalIter,1],
                       'Vp3':models[burn_in:FinalIter,2],
                       
                       #'Vp3':[models[i]['Vp'][2] for i in range(NMod)],
                       'Z1': models[burn_in:FinalIter,3],
                       'Z2': models[burn_in:FinalIter,4],
                      })
#Drop the burn-in samples 
#ModelsDF=ModelsDF.drop(range(1000))
true_vals = [3100,4470,6200,2000,4000]

prior_z = uniform(loc=1200,scale=5500).rvs(ModelsDF.shape[0])
prior_vp = uniform(loc=2100,scale=5300).rvs(ModelsDF.shape[0])
priors = [prior_vp]*3+[prior_z]*2
starts = [2700,4200,5700,2200,3700]
fig,ax=plt.subplots(nrows=5,ncols=1,figsize=(8,13))
for i,ax_cur,true_val,prior,start in zip(ModelsDF.columns,ax,true_vals,priors,starts):
    ModelsDF[i].hist(ax=ax_cur,bins=50,normed=True)
    ax_cur.hist(prior,bins=100,normed=True)
    ylim=ax_cur.get_ylim()
    ax_cur.plot([true_val,true_val],ylim, 'r')
    ax_cur.plot([start,start],ylim, 'g')
    
    ax_cur.legend(['True value','Init value','Hist of %s' % i, 'Prior hist'])

fig.savefig('MCMCHist.png')

ModelMatrix = ModelsDF.as_matrix()

pca_model = PCA().fit(ModelMatrix)
NewModelMatrix = pca_model.transform(ModelMatrix)
InvModel = pca_model.inverse_transform(NewModelMatrix)
filename = 'PCA3Layer.pkl'
_ = joblib.dump(pca_model, filename, compress=3)

kthin = 2

dr_array = model_file['dr_array'][range(0,ModelMatrix.shape[0],kthin)]
noise_std = np.apply_along_axis(np.std,1,dr_array)
noise_mean = np.apply_along_axis(np.mean,1,dr_array)

LL = LL[range(0,ModelMatrix.shape[0],kthin)]
ModelsDF=ModelsDF.ix[range(0,ModelMatrix.shape[0],kthin)]
ModelsDF.index=range(0,ModelsDF.shape[0])

ModelMatrix = ModelMatrix[range(0,ModelMatrix.shape[0],kthin),:]
# Plot velocity profiles:
z3=9000
lines_col=[]
Nd=50
Nv=300
profiles = np.zeros((Nd,ModelsDF.shape[0]))
velocities = np.linspace(800,7500,Nv)
depth_for_profile = np.linspace(0,8000,Nd)
hists=np.zeros((profiles.shape[0],velocities.shape[0]-1))
for i,rowSt in ModelsDF.iterrows():
    z1=rowSt.Z1
    z2=rowSt.Z2
    y=[0,z1,z1,z2,z2,z3]
    v1=rowSt.Vp1
    v2=rowSt.Vp2
    v3=rowSt.Vp3
    x=[v1,v1,v2,v2,v3,v3]
    f = interp1d(y,x,kind='zero')
    profiles[:,i]=f(depth_for_profile)
    lines_col.append([(v1,0),(v1,z1)])
               
    lines_col.append(         [(v1,z1),(v2,z1)])
    lines_col.append(       [(v2,z1),(v2,z2)])
    lines_col.append(         [(v2,z2),(v3,z2)])
    lines_col.append(         [(v3,z2),(v3,z3)])
for i in range(profiles.shape[0]):
    hists[i,:],_ = np.histogram(profiles[i,:],bins = velocities,normed=True)
X,Y=np.meshgrid(velocities,depth_for_profile)

fig2,ax_vel = pl.subplots(figsize=(6,8),ncols=2)
ax2=ax_vel[0]
ax3=ax_vel[1]
ax2.pcolormesh(X,Y,hists,vmin=0,vmax=hists.max()/6)
ax2.set_ylim((-0.1,8000))
ax2.set_xlim((500,8000))

ax2.invert_yaxis()


ax3.plot(x,y,'-r',alpha=0.0003)
print ' Adding collection'
lc=mc.LineCollection(lines_col,alpha=0.002)
ax3.add_collection(lc)
print 'Added collection'
ax3.set_ylim((-0.1,8000))
ax3.set_xlim((500,8000))
ax3.scatter(5000*np.ones(eqdf.shape[0]),eqdf.z,s=30,c = 'r')
ax3.invert_yaxis()
fig2.savefig('VelProfileMCMC.png',dpi=300)
#ax2.autoscale()
# Plot the chain elements history
fig4,ax4 = plt.subplots(figsize = (8,6))
ModelsDF.plot(ax=ax4,subplots=True)
fig4.savefig('TimeHistoryChain.png')
#fig2.savefig('VelProfileMCMC.png',dpi=300)

# Plot the datafit:
fig_data,ax_data = plt.subplots(figsize = (24,10))
ind_ordered = np.argsort(tp.flatten())

for ii,i in enumerate(ind_ordered):
    tp_slice = tp_array[:,i]
    tp_slice=tp_slice[(~np.isnan(tp_slice)) & (tp_slice > 0) ]
    hist_data,bins = np.histogram(tp_slice,bins = 150,normed=True)
    hist_data = np.vstack((hist_data,hist_data)).T
    bins = (bins[0:-1]+bins[1:])/2.0

    x,y = np.meshgrid([ii-0.45,ii+0.45],bins)
    ax_data.pcolormesh(x,y,hist_data,vmin=0,vmax=hist_data.max(),cmap=cm.jet,shading='gouraud',edgecolor=None)
    ax_data.scatter(ii,tp.flatten()[i],c='k',s=30)
ax_data.set_xlim([0,np.size(tp)])
ax_data.set_ylim([0.9,2.1])

fig_data.savefig('DataFit.png')
  

                                
covm=np.corrcoef(ModelMatrix.T)
plt.imshow(np.abs(covm),vmin=0.5,vmax=1,interpolation='None')


returnn

