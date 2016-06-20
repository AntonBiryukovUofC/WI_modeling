import obspy
import numpy as np
import re
import glob
locations = ["Row0Col0","Row0Col1"]
stations = ["0001","0002","0003"]
y=[]
NMoments = 600
for x in range(NMoments): y.append("M" + "%04d" % x)
    
    
moments = y
root_dir = "./AllTraces/"
list_mseed = glob.glob(root_dir + "*.mseed")
trace_for_stats= obspy.read("/home/geoanton/WI_modeling/Row0Col0/moment1/station0003/"+"B00101Z00.sac")

ObservationMatrix = np.empty([NMoments*2,trace_for_stats[0].stats.npts * len(stations)])
Y = np.empty([NMoments*2,1])
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
            observation = np.append(observation,temp_trace[0].data)
        ObservationMatrix[nrow,:] = observation
        Y[nrow,0] = LocationClass
        nrow+=1
       
np.savetxt("Observations.csv", ObservationMatrix, delimiter=",",fmt = "%.3e")
np.savetxt("ClassLabels.csv", Y, delimiter=",",fmt = "%d")