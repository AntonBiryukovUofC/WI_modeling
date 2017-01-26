# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import DoForwardModel
# For plotting / data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,uniform,norm



def ChangeModel(model,new_m):
    
    for l,new_zt,new_zb,new_vp in zip(model.layers(), new_m['Ztop'],new_m['Zbot'],new_m['Vp']) :
        l.mtop.vp=new_vp
        l.ztop=new_zt
        l.zbot=new_zb
    return model
        
#######################################################################

# Load the data which will be fitted 
data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
# Set up initial model:
Vinit=4000
proposal_width_vp = 500 # proposal width of the velocity
proposal_width_z = 1000
z1,z2=3000,5000
# model is V1,V2,V3,Z1,Z2 , Ztop =0 and Zbot=7000 are fixed values ( global top and bottom of the model)
model_vector = {'Vp':[Vinit,Vinit,Vinit],'Ztop':[0,z1,z2],'Zbot':[z1,z2,7010]}
current_m=model_vector
model =cake.load_model(('MCMCTest.nd')) # <--- True model for the forward simulation.

# Do initial forward model:
model=ChangeModel(model,model_vector)
sim_tp,_,_ = DoForwardModel(eqdf,stdf,model)
# Calculate its likelihood:
res_norm = np.dot(tp-sim_tp,np.dot(sigma_inv,tp-sim_tp))
likelihood_current = 1.0/(np.sqrt(2*np.pi)**(k/2)*sigma_det)* np.exp(-0.5*res_norm)
#######################################################################



#######################################################################
# Set up the distributions:
# Proposal - multivariate Normal with mean at current position and cov      
mean = model_vector['Vp'] + [z1,z2]
cov = [proposal_width_vp]*3 + [proposal_width_z]*2
proposal = multivariate_normal(mean,cov)
# Priors on interfaces and velocities:
prior_z = uniform(loc=1,scale=7000)
prior_vp = uniform(loc=1500,scale=6000)
# Noise on the arrivals :
t_noise = 0.045
k=tp.shape[0]
sigma=np.diag([t_noise]*k)
sigma_inv = np.linalg.inv(sigma)
sigma_det = np.linalg.det(sigma)
# Apply this noise on data:
tp +=norm(loc=0,scale=t_noise,random_state=123).rvs(tp.shape)
#######################################################################
# Calculate likelihood here for the proposed move:
# Calculate the proposed forward model :
proposed_m = proposal.rvs()
sim_tp = np.zeros_like(tp)
res_norm = np.dot(tp-sim_tp,np.dot(sigma_inv,tp-sim_tp))
likelihood_proposed = 1.0/(np.sqrt(2*np.pi)**(k/2)*sigma_det)* np.exp(-0.5*res_norm)




# Calculate prior probabilities for current_m and proposed_m:
# current
prior_current = prior_z.pdf(current_m['Ztop'][1:]).prod()*prior_vp.pdf(current_m['Vp']).prod()
# proposed
prior_proposed=prior_z.pdf(proposed_m[3,4]).prod()*prior_vp.pdf(proposed_m[0,1,2]).prod()








# Calculate the probability ratio:
p_current = likelihood_current*prior_current
p_proposal = likelihood_proposed*prior_proposed
p_accept = np.log(p_proposal) - np.log(p_current)
accept = np.random.rand() < p_accept
if accept:
    # We update the position
    model_vector = {'Vp':proposed_m[0,1,2],'Ztop':[0]+proposed_m[3,4],'Zbot':proposed_m[3,4]+[7010]}
    current_m = model_vector
    model=ChangeModel(model,model_vector)
    
    







#for l in model.layers():
        #print l.ztop


