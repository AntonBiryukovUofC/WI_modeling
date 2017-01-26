import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')
import matplotlib.ticker as plticker
import matplotlib.cm as cm
import glob
import os
import obspy
from MiscFunctions import getAmplitudeEnvelopeFeaturesReal
norm1=mpl.colors.Normalize(vmin=2,vmax=5)
m = cm.ScalarMappable(norm=norm1, cmap=cm.Accent)

loc = plticker.MultipleLocator(base=1000.0) # this locator puts ticks at regular intervals


eventsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekEvents.csv')
stationsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekStations.csv')
eventsSynth = pd.read_csv('SourcesWithPicks.csv')

Xref = stationsCatalog.X[3]
Yref = stationsCatalog.Y[3]
stationsCatalog.X-=Xref
stationsCatalog.Y-=Yref


eventsCatalog.X = eventsCatalog.X  - Xref
eventsCatalog.Y = eventsCatalog.Y  - Yref

w,h=8,8
fig,ax = plt.subplots(figsize = ((w,h)))
ax.scatter(x = eventsCatalog.X,y = eventsCatalog.Y)
ax.scatter(x = stationsCatalog.X,y = stationsCatalog.Y,c='g',s=40)
ax.scatter(x = eventsSynth.X,y = eventsSynth.Y,c='r',s=40)
eventsCatalogSub = eventsCatalog[(eventsCatalog.Y<7000) & (eventsCatalog.Y>4000) & (eventsCatalog.X>3000) & (eventsCatalog.X<4500)]
ax.scatter(x = eventsCatalogSub.X,y = eventsCatalogSub.Y,c='k')

#ax.text(x = stationsCatalog.X,y = stationsCatalog.Y,labels = stationsCatalog.Name)
ax.yaxis.set_major_locator(loc)
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(plticker.FormatStrFormatter('%2.1d'))
ax.yaxis.set_major_formatter(plticker.FormatStrFormatter('%2.1d'))
fig,ax = plt.subplots()
for i,row in eventsCatalogSub.iterrows():
    x = row[['PSta2', 'SSta2', 'PSta3',  'SSta3', 'PSta4', 'SSta4']]
    y= np.ones(6)*row['Depth (km)']
    ax.scatter(x,y,c= m.to_rgba(y[0]),s=20)
    
ax.set_title('Station P and S - picks')

eventsFeatures = np.zeros((eventsCatalogSub.shape[0],18))
k=0
for ie,row in eventsCatalogSub.iterrows():
    featureSet = np.zeros((3,6))
    for i,ist in zip([0,2,1],[4,3,2]):
        tracename = '/home/anton/WI_Models/FoxCreekMSEED/%04d/WSK%02dHHZ*.SAC' % (row.ID,ist)
        if len(glob.glob(tracename)) <1 or not(os.path.exists('/home/anton/WI_Models/FoxCreekMSEED/%04d/' % row.ID)):
            print 'Cant find folder / stream in \n %s ' % tracename
            featureSet[i,:] = featureSet[i-1,:]

        else:
            res = getAmplitudeEnvelopeFeaturesReal(traceName = tracename,st=row['PSta%d' % ist],fn=row['SSta%d' % ist],
                                starttime =obspy.UTCDateTime(row.UTCDate),endtime= obspy.UTCDateTime(row.UTCDate)+3.5 ,fmin=1,fmax=20)
            if not(res == None):
                featureSet[i,:] = res
                
            else:
                featureSet[i,:] = featureSet[i-1,:]
                print 'Weird startdate, imputing the values.. %s ' % tracename
    eventsFeatures[k,:] = featureSet.flatten()
    k+=1
synthFeatures= np.loadtxt("ObservationsPicksAndFeatures.csv", delimiter=",")[:,6:]
fig2,ax2 =  plt.subplots(nrows=18,ncols =1,figsize = (6,30))

label =['Kurtosis','Std','Mean','IntegEnvPS','EnergyRatio','ZC']*3
for i in range(18):
    ax2[i].hist(eventsFeatures[:,i],bins=30,color='r',alpha=0.5,normed=True)
    ax2[i].hist(synthFeatures[:,i],bins=30,color='b',alpha=0.5,normed=True)
    ax2[i].set_title(label[i])
fig2.tight_layout()
fig2.savefig('CompareFeats.png',dpi=200)






#clusterMid = eventsCatalog[eventsCatalog.X.abs()<2000]