# play witht the Pyrocko modules
from pyrocko import cake
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import os
from LocationsOnGrid import LocationsOnGridSmall
from MiscFunctions import GetPSArrivalRayTracing


model =cake.load_model('VpVs.nd')
SourcesCSV = pd.read_csv('sourcesDF.csv') # Read the source locations


_,_,_,stCoords = LocationsOnGridSmall(receiver_name='receiver.dat',NX=1,NY = 1,NZ =1) # Get the receiver locations
for iSt in range(stCoords.shape[0]):
    p_arrival =np.zeros(SourcesCSV.shape[0])
    s_arrival =np.zeros(SourcesCSV.shape[0])
    so_offset =np.zeros(SourcesCSV.shape[0])
    for index,row in SourcesCSV.iterrows():
        p_arrival[index],s_arrival[index],so_offset[index] = GetPSArrivalRayTracing(
                                                            sta_coords=stCoords[iSt,:],
                                                            eq_coords=np.array([row.X,row.Y,row.Z])
                                                            )
        print ' Done with station %d and event %d ' % (iSt,index)
    
    SourcesCSV['Psta%d' % (iSt+1)] = p_arrival
    SourcesCSV['Ssta%d' % (iSt+1)] = s_arrival
    SourcesCSV['Dsta%d' % (iSt+1)] = so_offset

SourcesCSV.to_csv('SourcesWithPicks.csv')


