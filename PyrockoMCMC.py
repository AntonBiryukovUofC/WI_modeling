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

data = np.load('ForwardDataMCMC.npz')        
tp,ts,so,stdf,eqdf = data['tp'],data['ts'],data['so'],data['stdf'],data['eqdf']
