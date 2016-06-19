import obspy
import numpy as np
import re
# Convolve the SAC files with the VerySmoothBump
from MiscFunctions import VerySmoothBump

import fnmatch
import os
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
LocationIdRe = re.compile('(Row.*?)/')

matchesOneChannel = [x for x in matches if channel in x]





for item in matchesOneChannel:
    i+=1
    temp= obspy.read(item)
    moment = MomentIdRe.search(item).group(1)
    station = StationIdRe.search(item).group(1)
    eq_location = LocationIdRe.search(item).group(1)
    
    temp[0].stats.moment = "M" + moment
    temp[0].stats.station = station
    temp[0].stats.EQ_location = eq_location
    temp[0].stats.location = eq_location
    temp[0].stats.network ="M" + moment 
    st+=temp

    if np.mod(i,500) == 0:
        print "%3.3f" % tuple([i/float(len(matchesOneChannel))])
st.sort(keys = ["network","station"])
    





returnn


t00 =0
endT = 3.3

t,y = VerySmoothBump(0.0,0,0.51,2,stWI[0].stats.delta)


stWI.trim(starttime = stWI[0].stats.starttime+t00, endtime = stWI[0].stats.starttime+endT+t00 )
stWI_Old = stWI.copy()

for trace in stWI:
        c = np.convolve(trace.data,y)
        c = c[0:len(trace)]
        trace.data = c

# Comb the signal a little bit
stWI.differentiate()

stWI.normalize()

stWI.sort()