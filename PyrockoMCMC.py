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
from scipy.stats import multivariate_normal



def ChangeModel(model,new_m):
    
    for l,new_zt,new_zb,new_vp in zip(model.layers(), new_m['Ztop'],new_m['Zbot'],new_m['Vp']) :
        l.mtop.vp=new_vp
        l.ztop=new_zt
        l.zbot=new_zb
    return model
        


data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
Vinit=4000
proposal_width_vp = 500 # proposal width of the velocity
proposal_width_z = 1000

z1,z2=3000,5000
# model is V1,V2,V3,Z1,Z2 , Ztop =0 and Zbot=7000 are fixed values ( global top and bottom of the model)
model_vector = {'Vp':[Vinit,Vinit,Vinit],'Ztop':[0,z1,z2],'Zbot':[z1,z2,7000]}
mean = model_vector['Vp'][:] + [z1,z2]
cov = [proposal_width_vp]*3 + [proposal_width_z]*2
      
proposal = multivariate_normal(mean,cov).rvs()






model =cake.load_model(('MCMCTest.nd')) # <--- True model for the forward simulation.

model=ChangeModel(model,model_vector)

for l in model.layers():
        print l.ztop


