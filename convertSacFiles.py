import obspy
import numpy as np
import re
# Convolve the SAC files with the VerySmoothBump
from MiscFunctions import VerySmoothBump

import fnmatch
import os
import subprocess
matches = []


for root, dirnames, filenames in os.walk('.'):
    for filename in fnmatch.filter(filenames, '*.sac'):
        matches.append(os.path.join(root, filename))

st = obspy.Stream()
i=1


# Work with one channel for now:
channel = "Z"+"00"

MomentIdRe = re.compile('moment(.*?)/')
StationIdRe = re.compile('station(.*?)/')
LocationIdRe = re.compile('(Class.*?)/')

matchesOneChannel = [x for x in matches if channel in x]





for item in matchesOneChannel:
    i+=1
    temp= obspy.read(item)
    moment = MomentIdRe.search(item).group(1)
    station = StationIdRe.search(item).group(1)
    eq_location = LocationIdRe.search(item).group(1)
    
    temp[0].stats.moment = "M" + ("%04d") % int(moment)
    temp[0].stats.station = station
    temp[0].stats.EQ_location = eq_location
    temp[0].stats.location = eq_location
    temp[0].stats.network =temp[0].stats.moment
    st+=temp

    if np.mod(i,500) == 0:
        print "%3.3f" % tuple([i/float(len(matchesOneChannel))])
st.sort(keys = ["network","station"])
    

t,y = VerySmoothBump(0.0,0,0.51,2,st[0].stats.delta)
print("Doing convolutions")
for trace in st:
        c = np.convolve(trace.data,y)
        c = c[0:len(trace)]
        trace.data = c

# Comb the signal a little bit
st.differentiate()
st.taper(type= "cosine",max_percentage=0.05)
st.normalize()

all_trace_dir = "./AllTraces/"
subprocess.call("mkdir "+ all_trace_dir, shell = True)
print("Saving traces into separate files with proper names")
for trace in st:
    trace.write(all_trace_dir +trace.stats.moment + "_station_"+trace.stats.station+"_location_"+trace.stats.location+"_channel_"+trace.stats.channel+ ".mseed",format="MSEED")



