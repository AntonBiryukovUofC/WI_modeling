import obspy
import numpy as np
import re
import glob
from MiscFunctions import circshift
import pandas as pd
#Add some variations with t_ref

SourcesCSV = pd.read_csv('SourcesWithPicks.csv')
NClass=SourcesCSV.shape[0]
y=[]
# perturbation for ref time:
shift = 10


for x in range(NClass): y.append("Class" + "%03d" % x)

locations = y
stations = ["0001","0002","0003"]
y=[]
NMoments = 150
for x in range(NMoments): y.append("M" + "%04d" % x)
    
    
moments = y
root_dir = "./AllTraces/"
list_mseed = glob.glob(root_dir + "*.mseed")
trace_for_stats= obspy.read("./Class000/moment1/station0003/"+"B00101Z00.sac")

ObservationMatrix = np.empty([NMoments*len(locations),trace_for_stats[0].stats.npts * len(stations)])
ObservationMatrixPicks = np.empty([NMoments*len(locations),2 * len(stations)])
ErrP = 0.0
ErrS = 0.0
Y = np.empty([NMoments*len(locations),1])
LocationClass = -1
nrow = 0
for loc in locations:
    LocationClass +=1
    print("Processing location class " + loc + " == " + str(LocationClass))
    for moment in moments:
        print("Processing moment " + moment)
        observation = np.array([])
        for sta in stations:
            part_of_name =moment +  "_station_"+sta+"_location_"+loc
            matches = [x for x in list_mseed if part_of_name in x]
            temp_trace = obspy.read(matches[0])
            ind = np.random.randint(low=-shift,high = shift)
            temp_trace = circshift(temp_trace,ind)
            temp_trace.taper(type= "cosine",max_percentage=0.05)

            observation = np.append(observation,temp_trace[0].data)
        ObservationMatrix[nrow,:] = observation
        rowSource = SourcesCSV.ix[SourcesCSV.Class == LocationClass]
        Psta1 = rowSource.Psta1 +  ErrP*np.random.randn(1)
        Psta2 = rowSource.Psta2 +  ErrP*np.random.randn(1)
        Psta3 = rowSource.Psta3 +  ErrP*np.random.randn(1)
        
        Ssta1 = rowSource.Ssta1 +  ErrS*np.random.randn(1)
        Ssta2 = rowSource.Ssta2 +  ErrS*np.random.randn(1)
        Ssta3 = rowSource.Ssta3 +  ErrS*np.random.randn(1)
        pickRow = np.array([Psta1.values,Psta2.values,Psta3.values,Ssta1.values,Ssta2.values,Ssta3.values])
        pickRow.shape=(6,)
        ObservationMatrixPicks[nrow,:] = pickRow
        
        Y[nrow,0] = LocationClass
        nrow+=1
       
np.savetxt("Observations.csv", ObservationMatrix, delimiter=",",fmt = "%.3e")
np.savetxt("ClassLabels.csv", Y, delimiter=",",fmt = "%d")
np.savetxt("ObservationsPicks.csv", ObservationMatrixPicks, delimiter=",",fmt = "%.3e")
