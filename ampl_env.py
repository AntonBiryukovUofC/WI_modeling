
import obspy 
import numpy as np
from obspy.signal.filter import envelope
from scipy.integrate import simps
from scipy.stats import kurtosis
trace = obspy.read('/home/anton/WI_Models/AllTraces/M0055_station_0003_location_Class036_channel_Z.mseed')
#data_envelopeM = obspy.signal.filter.envelope(trace.data)
TraceCopy  = trace[0].copy()
envTrace = trace[0].copy()
st=1.2
fn = 2.2
envTrace.data = envelope(TraceCopy.data)
# Feature 0 : peakedness 
KurtosisEnvelopeDiff =kurtosis(np.diff(envTrace.data))
# Feature 1
StdEnvelope = envTrace.data.std()
# Feature 2
MeanEnvelope = envTrace.data.mean()


envTrace.trim(starttime = trace[0].stats.starttime + st,endtime = trace[0].stats.starttime + fn)
TraceCopy.trim(starttime = trace[0].stats.starttime + st,endtime = trace[0].stats.starttime + fn)
#Integrate the envelope in the region:
# Feature 3
EnvelopeIntegral = simps(y = envTrace.data,x = envTrace.times()) / (envTrace.stats.endtime - envTrace.stats.starttime)
EnergyInTheSubTrace = simps(y = TraceCopy.data**2,x = TraceCopy.times())
EnergyWholeTrace = simps(y = trace[0].data**2,x = trace[0].times())

#Feature 4
EnergyRatio = EnergyInTheSubTrace/EnergyWholeTrace
#Feature 5 
zc= np.sign( np.diff(envTrace.data) )  
zc[zc==0] = -1     # replace zeros with -1  
zcFeature = np.where(np.diff(zc))[0].shape[0] 


envTrace.plot()


