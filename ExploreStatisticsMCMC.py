# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
from MiscFunctions import DoForwardModel,GetPSArrivalRayTracingMC,MakeModel
# For plotting / data wrangling
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal,uniform,norm
import time
from sklearn.externals import joblib        
#######################################################################
np.random.seed(1234) # set the seed
# Load the data which will be fitted 
def getJitRange(n,loc,scale):
    a=np.arange(n)
    jitter = norm(loc=loc,scale=scale).rvs(a.shape[0])
    a=a+jitter
    return a
N=30
x,y,z = 2*getJitRange(N,loc=0,scale=0.2),getJitRange(N,loc=0,scale=0.2),getJitRange(N,loc=0,scale=0.2)
Xtrain = np.stack((x,y)).T
pca_vec = PCA(n_components=2).fit(Xtrain)
newXtrain = pca_vec.transform(Xtrain)
xnew=newXtrain[:,0]
ynew=newXtrain[:,1]
#fig,ax = plt.subplots(figsize=(10,10),nrows=2)
#ax[0].scatter(x,y)
#ax[0].axis('equal')
#ax[1].scatter(xnew,ynew)
#ax[1].axis('equal')
x_test = np.array([[30,40,50,-20],[0,0,0,0]]).T
x_inv = pca_vec.inverse_transform(x_test)

model_file = np.load('models_PCA_First.npz')
models=model_file['models']
NMod=models.shape[0]
ModelsDF=pd.DataFrame({'Vp1':[models[i]['Vp'][0] for i in range(NMod)],
                       'Vp2':[models[i]['Vp'][1] for i in range(NMod)],
                       'Vp3':[models[i]['Vp'][2] for i in range(NMod)],
                       
                       #'Vp3':[models[i]['Vp'][2] for i in range(NMod)],
                       'Z1': [models[i]['Ztop'][1] for i in range(NMod)],
                       'Z2': [models[i]['Ztop'][2] for i in range(NMod)],
                      })
ModelMatrix=  ModelsDF.as_matrix()
                                
covm=np.corrcoef(ModelMatrix.T)

pca_model = PCA().fit(ModelMatrix)
NewModelMatrix = pca_model.transform(ModelMatrix)
InvModel = pca_model.inverse_transform(NewModelMatrix)
filename = 'PCA3Layer.pkl'
_ = joblib.dump(pca_model, filename, compress=3)









#returnn
data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
eqdf = pd.DataFrame(data=eqdf,columns=['x','y','z'])
stdf = pd.DataFrame(data=stdf,columns=['x','y','z'])
model_file = np.load('models_PCA_First.npz')
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
true_vals = [3100,4470,6200,2000,4000]

prior_z = uniform(loc=1,scale=7000).rvs(0.5*ModelsDF.shape[0])
prior_vp = uniform(loc=1500,scale=6000).rvs(0.5*ModelsDF.shape[0])
priors = [prior_vp]*3+[prior_z]*2
starts = [4000,4000,5000,3000,5000]


#fig_cov,ax_cov=plt.subplots(nrows=5,ncols=5)

ModelMatrix=  ModelsDF.as_matrix()
                                
covm=np.corrcoef(ModelMatrix.T)
#PCModel = PCA(n_components=5).fit(ModelMatrix)






#g = sns.pairplot(ModelsDF, kind="reg")
plt.imshow(np.abs(covm),vmin=0.5,vmax=1,interpolation='None')

#g = sns.PairGrid(ModelsDF)
#g.map(sns.kdeplot)

#fig.savefig('TestMCMC.png',dpi=300)
returnn

