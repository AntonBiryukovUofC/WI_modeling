import numpy as np
from emcee import PTSampler
import emcee
import pandas as pd
from MiscFunctions import DoForwardModel_MyTracer

from scipy.stats import multivariate_normal,uniform,norm,randint


def logprior(theta,prior_z,prior_vp,nLayers):

    prior = prior_z.pdf(theta[nLayers:nLayers+nLayers-1]).prod() * \
            prior_vp.pdf(theta[0:nLayers]).prod()
    return np.log(prior)



def loglike(theta, x, tp, sigma_inv,log_sigma_det,nLayers):
    #m, b, lnf = theta
    #model = m * x + b
    
    eqdf=x[0]
    Neq=eqdf.shape[0]
    stdf=x[1]
    Nst=stdf.shape[0]
    vels_new = theta[0:nLayers]
    dz = theta[nLayers:nLayers+nLayers-1]
    depths_new = np.cumsum(dz)
    #print vels_new
    #print depths_new
    print 'Forward Model called '
    sim_tp,_ = DoForwardModel_MyTracer(eqdf,stdf,vels_new,depths_new)
    dr=tp.flatten()-sim_tp.flatten()
    res_norm = np.dot(dr,
                  np.dot(sigma_inv,dr))
    log_likelihood = np.log(1.0/(np.sqrt(2*np.pi)**(Neq*Nst))) - log_sigma_det + (-0.5*res_norm)
    #inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    #return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return log_likelihood

def logprob(theta, x, tp, sigma_inv,log_sigma_det,nLayers,priors):
    lp = logprior(theta,priors[0],priors[1],nLayers)
    if not np.isfinite(lp):
        return -np.inf
    ll = loglike(theta, x, tp, sigma_inv,log_sigma_det,nLayers)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
    

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


# Model Priors
#Use z as thickness priors :
prior_z = uniform(loc=800,scale=5500)
prior_vp = uniform(loc=2100,scale=5300)




# Data noise
t_noise=0.08
sigma=np.diag([t_noise**2]*Neq*Nst)
sigma_inv = np.linalg.inv(sigma)
sigma_det = np.linalg.det(sigma)

sign,log_sigma_det = np.linalg.slogdet(sigma)
nwalkers = 12
ndim=5
x=[eqdf,stdf]
nLayers=3

# Set up initial model:
init_theta=np.array([2700,4200,5700,2200,3700])
init_pos = [init_theta + 100*np.random.randn(ndim) for i in range(nwalkers)]
priors = [prior_z,prior_vp]
sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(x,tp, sigma_inv,log_sigma_det,nLayers,priors)) 
sampler.run_mcmc(init_pos, 500)

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

import corner
fig = corner.corner(samples, labels=["V1", "V2", "V3","Z1","Z2"],
                      truths= [3100,4470,6200,2000,2000])


    