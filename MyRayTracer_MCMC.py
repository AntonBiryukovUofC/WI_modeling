# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import DoForwardModel_MyTracer
# For plotting / data wrangling
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,uniform,norm,randint
from sklearn.externals import joblib        
from sklearn.decomposition import PCA


    


Likelihood_turn_off =False
#######################################################################
np.random.seed(323232) # set the seed
# Load the data which will be fitted 
mname = 'MCMCTest'
data = np.load('Forward%s.npz' % mname)        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
eqdf = pd.DataFrame(data=eqdf,columns=['x','y','z'])
stdf = pd.DataFrame(data=stdf,columns=['x','y','z'])

# Noise on the arrivals :
# Apply this noise on data:                
t_noise = 0.08
Neq=tp.shape[0]
Nst=tp.shape[1]
sigma=np.diag([t_noise**2]*Neq*Nst)
sigma_inv = np.linalg.inv(sigma)
sigma_det = np.linalg.det(sigma)

sign,log_sigma_det = np.linalg.slogdet(sigma)
tp_true = tp.copy()
tp +=norm(loc=0,scale=t_noise).rvs(tp.shape)

# Set up initial model:
vLayers=np.array([2700,4200,5700])


proposal_width_vp = 100 # proposal width of the velocity
proposal_width_z = 80
zLayers=np.array([2200,3700])
   # Priors on interfaces and velocities:
prior_z = uniform(loc=1200,scale=5500)
prior_vp = uniform(loc=2100,scale=5300)
# model is V1,V2,V3,Z1,Z2 , Ztop =0 and Zbot=7000 are fixed values ( global top and bottom of the model)
#model =cake.load_model(('MCMCTest.nd')) # <--- True model for the forward simulation.



vels = vLayers
depths=zLayers
# Do initial forward model:
sim_tp,so = DoForwardModel_MyTracer(eqdf,stdf,vels,depths)
# Calculate its likelihood:
dr=tp.flatten()-sim_tp.flatten()
res_norm = np.dot(dr,
                  np.dot(sigma_inv,dr))

dr_true=tp_true.flatten()-tp.flatten()

res_norm_true = np.dot(dr_true,
                  np.dot(sigma_inv,dr_true))

#returnn
log_likelihood_true = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst))) - log_sigma_det + (-0.5*res_norm_true)
log_likelihood_current = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst))) - log_sigma_det + (-0.5*res_norm)
#######################################################################
k_accept=0
MCMCiter = 80000
MCMCiter +=1

# Every NPCA update the covariance
NPCA=2000

proposed_array=np.zeros((MCMCiter,len(vLayers)+len(zLayers)))
ar=0.32
LL=np.zeros(MCMCiter)
# open the PC component pickle, analyse the widths for the proposal
filename = 'PCA3Layer.pkl'
pca_model = joblib.load(filename)





models = np.zeros((MCMCiter,len(vLayers)+len(zLayers)))
dr_array =np.zeros((MCMCiter,sim_tp.flatten().shape[0]))
tp_array =np.zeros((MCMCiter,sim_tp.flatten().shape[0]))

dm = np.zeros(MCMCiter)


PCA_idx_sampler = randint(low=0,high=len(vLayers)+len(zLayers))

frac_of_sigma = 0.15
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
    #else:
        
     #   proposal_width_vp=proposal_width_vp*1
      #  proposal_width_z=proposal_width_z*1
       # print ' DEcreased the widths !'
#    proposal_width_vp = 100 # proposal width of the velocity
 #   proposal_width_z = 80
       #
       
       
    if i >NPCA+10:   
        # Get the current mean of the proposal.
        mean = np.hstack((vels,depths))
        # Express it in the PC axes
        mean_PCA = pca_model.transform(mean.reshape(1,-1)).squeeze()
        #cov = [proposal_width_vp**2]*len(vLayers) + [proposal_width_z**2]*len(zLayers)
        
        # Pick by random a PC to perturb
        index_PCA = PCA_idx_sampler.rvs()
        indexer = np.zeros(len(vLayers)+len(zLayers))
        indexer[index_PCA] = 1
        
        
        # Set the covariance of the proposal as a fraction of the eigenvalue
        
        cov_PCA = pca_model.explained_variance_[index_PCA] * frac_of_sigma**2
        
        # Sample the picked PC :
        proposal = norm(0,cov_PCA).rvs()
        sample_proposed_pca = mean_PCA + indexer * proposal
        # Transform back to initial axes:
        sample_proposed = pca_model.inverse_transform(sample_proposed_pca.reshape(1,-1)).squeeze()
    else:
        print 'Sampling from non-PCA..'
        mean = list(vels) + list(depths)
        cov = [proposal_width_vp**2]*3 + [proposal_width_z**2]*2
        sample_proposed = multivariate_normal(mean,cov).rvs()

    
    # Sort the depths
    
    
    
    sample_proposed[len(vLayers):]=np.sort(sample_proposed[len(vLayers):]) 
    
    # SAve for debugging
    proposed_array[i,:]=sample_proposed
    
        
    proposed_m = sample_proposed
    prior_new=prior_z.pdf(proposed_m[len(vLayers):len(vLayers)+len(zLayers)]).prod()*prior_vp.pdf(proposed_m[0:len(vLayers)]).prod()
    if (prior_new == 0):
        # Zero prior, not moving , adding the old model
        models[i,0:len(vLayers)]=vels
        models[i,len(vLayers):len(vLayers)+len(zLayers)]=depths
        LL[i] = log_likelihood_current

        print 'Zero Prior of the proposed move!'
        continue
  
    # Get a new model for the proposed move.
    vels_new = proposed_m[0:len(vLayers)]
    depths_new = proposed_m[len(vLayers):]
    
    sim_tp,_ = DoForwardModel_MyTracer(eqdf,stdf,vels_new,depths_new)
    #print ' Forward model for %d sample done ' % i
    dr=tp.flatten()-sim_tp.flatten()
 
    dr_array[i,:]=dr
    tp_array[i,:]=sim_tp.flatten()
    res_norm = np.dot(dr,
                  np.dot(sigma_inv,dr))
    #  likelihood_proposed = 1.0/(np.sqrt(2*np.pi)**(Neq*Nst/2)*sigma_det)* np.exp(-0.5*res_norm)
    log_likelihood_proposed = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst))) - log_sigma_det + (-0.5*res_norm)
    
  

    # Calculate prior probabilities for current_m and proposed_m:
    # current
    prior_cur = prior_z.pdf(depths).prod()*prior_vp.pdf(vels).prod()
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
        vels=vels_new
        depths=depths_new
        
        
        log_likelihood_current = log_likelihood_proposed
    # Add the model into i-th place :
    models[i,0:len(vLayers)]=vels
    models[i,len(vLayers):len(vLayers)+len(zLayers)]=depths
    
    LL[i] = log_likelihood_current

    if (i % NPCA) ==0 and (i>0):
        np.savez('models_PCA.npz',models=models,LL=LL,proposed_array=proposed_array,k_accept=k_accept,iter=i,dr_array=dr_array,LL_true = log_likelihood_true,tp_array=tp_array)
        ModelMatrix = models[i-NPCA:i]
        pca_model = PCA().fit(ModelMatrix)
       

    if i> 50:
        ar=1.0*k_accept/(i)

        if ar>0.4:
            frac_of_sigma=frac_of_sigma*1.1
        #proposal_width_vp=proposal_width_vp*1
       # proposal_width_z=proposal_width_z*1
            print ' Increased the widths !'
        elif ar<0.2:
            frac_of_sigma=frac_of_sigma*0.9
       # proposal_width_vp = 100 # proposal width of the velocity
       # proposal_width_z = 80
            print ' Decreased the widths!'
    

    #returnn

