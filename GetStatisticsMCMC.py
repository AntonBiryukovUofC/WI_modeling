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
                       'Z1': [models[i]['Ztop'][1] for i in range(NMod)],
                       'Z2': [models[i]['Ztop'][2] for i in range(NMod)],
                      })
                      

returnn

