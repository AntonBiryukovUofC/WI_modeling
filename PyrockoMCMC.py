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
from sklearn.externals import joblib        



Likelihood_turn_off =False
#######################################################################
np.random.seed(1234) # set the seed
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
tp +=norm(loc=0,scale=t_noise).rvs(tp.shape)

# Set up initial model:
vLayers=[2700,4200,5700]


proposal_width_vp = 100 # proposal width of the velocity
proposal_width_z = 80
zLayers=[2200,3700]
   # Priors on interfaces and velocities:
prior_z = uniform(loc=1500,scale=3500)
prior_vp = uniform(loc=2500,scale=4300)
# model is V1,V2,V3,Z1,Z2 , Ztop =0 and Zbot=7000 are fixed values ( global top and bottom of the model)
model_vector = {'Vp':vLayers,'Ztop':[0] + zLayers,'Zbot':zLayers + [9000]}
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
log_likelihood_current = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst))) - log_sigma_det + (-0.5*res_norm)
#######################################################################
k_accept=0
MCMCiter = 40000
MCMCiter +=1

proposed_array=np.zeros((MCMCiter,len(vLayers)+len(zLayers)))
ar=0.32
LL=[]
# open the PC component pickle, analyse the widths for the proposal
filename = 'PCA3Layer.pkl'
pca_model = joblib.load(filename)


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
    #if ar>0.4:
       # proposal_width_vp=proposal_width_vp*1
       # proposal_width_z=proposal_width_z*1
       # print ' INcreased the widths !'
    #elif ar>=0.3:
       # proposal_width_vp = 100 # proposal width of the velocity
       # proposal_width_z = 80
       # print ' The widths are back to normal!'
    #else:
     #   proposal_width_vp=proposal_width_vp*1
      #  proposal_width_z=proposal_width_z*1
       # print ' DEcreased the widths !'
#    proposal_width_vp = 100 # proposal width of the velocity
 #   proposal_width_z = 80
       #
    # Get the current mean.
    mean = np.array(current_m['Vp'] + current_m['Ztop'][1:])
    # Express it in the PC axes
    mean_PCA = pca_model.transform(mean.reshape(1,-1)).squeeze()
    #cov = [proposal_width_vp**2]*len(vLayers) + [proposal_width_z**2]*len(zLayers)
    # Set the covariance of the proposal as a fraction of the eigenvalue
    cov_PCA = pca_model.explained_variance_ * frac_of_sigma**2
    
    proposal = multivariate_normal(mean_PCA,cov_PCA)
    # Sample from the proposal
    sample_proposed_pca = proposal.rvs()
    # Transform back to initial axes:
    sample_proposed = pca_model.inverse_transform(sample_proposed_pca.reshape(1,-1)).squeeze()
    sample_proposed[len(vLayers):]=np.sort(sample_proposed[len(vLayers):]) 
    # SAve for debugging
    proposed_array[i,:]=sample_proposed
    
    
    proposed_m = list(sample_proposed)
    prior_new=prior_z.pdf(proposed_m[len(vLayers):len(vLayers)+len(zLayers)]).prod()*prior_vp.pdf(proposed_m[0:len(vLayers)]).prod()
    if (prior_new == 0):
        models.append(model_vector)
        print 'Zero Prior of the proposed move!'
        continue
    # Convert to np array
    model_vector_new = {'Vp':proposed_m[0:len(vLayers)],
                        'Ztop':[0] + proposed_m[len(vLayers):],
                        'Zbot':proposed_m[len(vLayers):] + [9000]}

    
    model_new=MakeModel(model_vector_new) # Get a new model for the proposed move.
    sim_tp,_,_ = DoForwardModel(eqdf,stdf,model_new)
    print ' Forward model for %d sample done ' % i
    dr=tp.flatten()-sim_tp.flatten()
    res_norm = np.dot(dr,
                  np.dot(sigma_inv,dr))
    #  likelihood_proposed = 1.0/(np.sqrt(2*np.pi)**(Neq*Nst/2)*sigma_det)* np.exp(-0.5*res_norm)
    log_likelihood_proposed = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst))) - log_sigma_det + (-0.5*res_norm)



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
    LL.append(log_likelihood_current)
    if (i % 200) ==0:
        np.savez('models_PCA.npz',models=models,LL=LL,proposed_array=proposed_array,k_accept=k_accept)

    if i>50:
        ar=1.0*k_accept/i
    LL.append(log_likelihood_current)
for l in model.layers():
        print l.mbot.vp


