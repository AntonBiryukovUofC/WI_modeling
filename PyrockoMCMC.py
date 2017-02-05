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




Likelihood_turn_off =False
#######################################################################
np.random.seed(1234) # set the seed
# Load the data which will be fitted 
data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
eqdf = pd.DataFrame(data=eqdf,columns=['x','y','z'])
stdf = pd.DataFrame(data=stdf,columns=['x','y','z'])

# Noise on the arrivals :
# Apply this noise on data:                
t_noise = 0.04
Neq=tp.shape[0]
Nst=tp.shape[1]
sigma=np.diag([t_noise**2]*Neq*Nst)
sigma_inv = np.linalg.inv(sigma)
sigma_det = np.linalg.det(sigma)

sign,log_sigma_det = np.linalg.slogdet(sigma)
tp +=norm(loc=0,scale=t_noise).rvs(tp.shape)

# Set up initial model:
Vinit=4000
proposal_width_vp = 400 # proposal width of the velocity
proposal_width_z = 400
z1,z2=3000,5000
   # Priors on interfaces and velocities:
prior_z = uniform(loc=1,scale=7000)
prior_vp = uniform(loc=1500,scale=6000)
# model is V1,V2,V3,Z1,Z2 , Ztop =0 and Zbot=7000 are fixed values ( global top and bottom of the model)
model_vector = {'Vp':[Vinit,Vinit,Vinit],'Ztop':[0,z1,z2],'Zbot':[z1,z2,9000]}
current_m=model_vector
#model =cake.load_model(('MCMCTest.nd')) # <--- True model for the forward simulation.
model=MakeModel(model_vector)



            
models=[]
# Do initial forward model:
sim_tp,_,_ = DoForwardModel(eqdf,stdf,model)
# Calculate its likelihood:
dr=tp.flatten()-sim_tp.flatten()
res_norm = np.dot(dr,
                  np.dot(sigma_inv,dr))
#returnn
log_likelihood_current = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst/2))) - log_sigma_det + (-0.5*res_norm)
#######################################################################
k_accept=0
MCMCiter = 400
MCMCiter +=1

proposed_array=np.zeros((MCMCiter,5))

for i in range(MCMCiter):
#######################################################################
# Set up the distributions:
# Proposal - multivariate Normal with mean at current position and cov      
   # mean = model_vector['Vp'] + [z1,z2]
    #cov = [proposal_width_vp**2]*3 + [proposal_width_z**2]*2
    #proposal = multivariate_normal(mean,cov)
 

#######################################################################
# Calculate likelihood here for the proposed move:
# Calculate the proposed forward model :
    mean = current_m['Vp'] + current_m['Ztop'][1:]
    cov = [proposal_width_vp**2]*3 + [proposal_width_z**2]*2
          
    proposal = multivariate_normal(mean,cov)
    sample_proposed = proposal.rvs()
    sample_proposed[3:]=np.sort(sample_proposed[3:]) 
    # SAve for debugging
    proposed_array[i,:]=sample_proposed
    
    
    proposed_m = list(sample_proposed)
    prior_new=prior_z.pdf(proposed_m[3:5]).prod()*prior_vp.pdf(proposed_m[0:3]).prod()
    if (prior_new == 0):
        models.append(model_vector)
        print 'Zero Prior of the proposed move!'
        continue
        

    model_vector_new = {'Vp':proposed_m[0:3],
                        'Ztop':[0] + proposed_m[3:],
                        'Zbot':proposed_m[3:] + [7010]}

    
    model_new=MakeModel(model_vector_new) # Get a new model for the proposed move.
    sim_tp,_,_ = DoForwardModel(eqdf,stdf,model_new)
    print ' Forward model for %d sample done ' % i
    dr=tp.flatten()-sim_tp.flatten()
    res_norm = np.dot(dr,
                  np.dot(sigma_inv,dr))
    #  likelihood_proposed = 1.0/(np.sqrt(2*np.pi)**(Neq*Nst/2)*sigma_det)* np.exp(-0.5*res_norm)
    log_likelihood_proposed = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst/2))) - log_sigma_det + (-0.5*res_norm)



    # Calculate prior probabilities for current_m and proposed_m:
    # current
    prior_cur = prior_z.pdf(current_m['Ztop'][1:]).prod()*prior_vp.pdf(current_m['Vp']).prod()
    # proposed
    if Likelihood_turn_off:
        log_likelihood_proposed=log_likelihood_current
    # Calculate the probability ratio:
    p_accept = np.log(prior_new)+log_likelihood_proposed - (np.log(prior_cur)+log_likelihood_current)
    print p_accept
    
    accept = ( (np.log(np.random.rand()) - p_accept) < 0 )
    if accept:
        # We update the position
        k_accept+=1
        print ' Proposal accepted %d out of %d ' %(k_accept,i)
        model_vector = model_vector_new
        current_m = model_vector
        model=model_new
        log_likelihood_current = log_likelihood_proposed
    models.append(model_vector)
    if (i % 200) ==0:
        np.savez('models_a.npz',models=models)




for l in model.layers():
        print l.mbot.vp


