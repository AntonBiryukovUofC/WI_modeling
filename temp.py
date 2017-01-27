# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import DoForwardModel,GetPSArrivalRayTracingMC
# For plotting / data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,uniform,norm



def ChangeModel(model,new_m):
    
    for l,new_zt,new_zb,new_vp in zip(model.layers(), new_m['Ztop'],
                                      new_m['Zbot'],new_m['Vp']) :
        l.mtop.vp=new_vp
        l.mtop.rho=0.31*new_vp**(0.25)*1000
        l.ztop=new_zt
        l.zbot=new_zb
    return model
    
def ChangeModelFake(model,new_m):
    
    for l,new_zt,new_zb,new_vp in zip(model.layers(), new_m['Ztop'],
                                      new_m['Zbot'],new_m['Vp']) :
        l.mtop.vp=3500.0
        l.mtop.rho=1.4*1000
        l.mbot.vp=3500.0
        l.mbot.rho=1.4*1000
        l.ztop=new_zt
        l.zbot=new_zb
    return model    
        
#######################################################################
np.random.seed(1234) # set the seed
# Load the data which will be fitted 
data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
eqdf = pd.DataFrame(data=eqdf,columns=['x','y','z'])
stdf = pd.DataFrame(data=stdf,columns=['x','y','z'])

# Noise on the arrivals :
# Apply this noise on data:                
t_noise = 0.035
k=tp.shape[0]
sigma=np.diag([t_noise]*k)
sigma_inv = np.linalg.inv(sigma)
sigma_det = np.linalg.det(sigma)
tp +=norm(loc=0,scale=t_noise).rvs(tp.shape)

# Set up initial model:
Vinit=4000
proposal_width_vp = 500 # proposal width of the velocity
proposal_width_z = 1000
z1,z2=3000.0,5000.0
   # Priors on interfaces and velocities:
prior_z = uniform(loc=1,scale=7000)
prior_vp = uniform(loc=1500,scale=6000)
# model is V1,V2,V3,Z1,Z2 , Ztop =0 and Zbot=7000 are fixed values ( global top and bottom of the model)
model_vector = {'Vp':[Vinit,Vinit,Vinit],'Ztop':[0.0,z1,z2],'Zbot':[z1,z2,7010.0]}
current_m=model_vector
model =cake.load_model(('MCMCTest.nd')) # <--- True model for the forward simulation.

model1=cake.load_model(('MCMCTest1.nd')) 



