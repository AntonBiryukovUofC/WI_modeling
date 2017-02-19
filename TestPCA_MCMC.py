# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import DoForwardModel,GetPSArrivalRayTracingMC,MakeModel
# For plotting / data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,uniform,norm
import time

        
#######################################################################
np.random.seed(1234) # set the seed
# Load the data which will be fitted 
data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
eqdf = pd.DataFrame(data=eqdf,columns=['x','y','z'])
stdf = pd.DataFrame(data=stdf,columns=['x','y','z'])
model_file = np.load('models_a.npz')
models=model_file['models']
NMod=models.shape[0]
ModelsDF=pd.DataFrame({'Vp1':[models[i]['Vp'][0] for i in range(NMod)],
                       'Vp2':[models[i]['Vp'][1] for i in range(NMod)],
                       'Vp3':[models[i]['Vp'][2] for i in range(NMod)],
                       
                       #'Vp3':[models[i]['Vp'][2] for i in range(NMod)],
                       'Z1': [models[i]['Ztop'][1] for i in range(NMod)],
                       'Z2': [models[i]['Ztop'][2] for i in range(NMod)],
                      })
#Drop the burn-in samples 
#ModelsDF=ModelsDF.drop(range(1000))




returnn

true_vals = [3100,4470,6200,2000,4000]

prior_z = uniform(loc=1,scale=7000).rvs(0.5*ModelsDF.shape[0])
prior_vp = uniform(loc=1500,scale=6000).rvs(0.5*ModelsDF.shape[0])
priors = [prior_vp]*3+[prior_z]*2
starts = [4000,4000,5000,3000,5000]
fig,ax=plt.subplots(nrows=5,ncols=1,figsize=(8,13))
for i,ax_cur,true_val,prior,start in zip(ModelsDF.columns,ax,true_vals,priors,starts):
    ModelsDF[i].hist(ax=ax_cur,bins=50)
    ax_cur.hist(prior,bins=100)
    ylim=ax_cur.get_ylim()
    ax_cur.plot([true_val,true_val],ylim, 'r')
    ax_cur.plot([start,start],ylim, 'g')
    
    ax_cur.legend(['True value','Init value','Hist of %s' % i, 'Prior hist'])


fig.savefig('TestMCMC.png',dpi=300)

