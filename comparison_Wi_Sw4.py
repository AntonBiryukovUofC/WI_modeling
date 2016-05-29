import obspy
import glob
import re
import numpy as np
import matplotlib.pyplot as plt 

# Get the traces from SW4
from MiscFunctions import rickerInt, VerySmoothBump

# Change X and Y for SW4 Plotting!!!!!!!!!!!!
root_dir = "./Model_Default/"
station="3"
list_sac =  glob.glob(root_dir +'*_'+station+ ".*");
list_sac.sort();
stN=obspy.Stream();
stWI=obspy.Stream();
for file in list_sac:
    st_temp=obspy.read(file)
    m = re.search('.[xyz]', file)
    if m:
        found = m.group(0)
        
    st_temp[0].stats.network = found
    st_temp[0].stats.station = station
    st_temp[0].stats.channel = found

    stN+=st_temp
#stN.filter("bandpass",freqmin = 0.0005,freqmax = 15)
#stN = stN.sort(reverse=True)
stN.normalize()

#stN.plot(type= 'relative')


num_sta = station
list_WI_sac =  glob.glob("./station000"+num_sta+"/*.sac")

#list_WI_sac =  glob.glob("/home/anton/WI_Models/"+"*.sac")
for file in list_WI_sac:
    stWI+=obspy.read(file)
    
#stN.filter("bandpass",freqmin = 0.0005,freqmax = 5)
#stWI.integrate()
#stWI.filter("bandpass",freqmin = 0.0005,freqmax = 8)

# Signal to convolve with
#t,y = rickerInt(0.45,0,1.1,2,stWI[0].stats.delta)
t00 =0
t,y = VerySmoothBump(0.0,0,0.51,2,stWI[0].stats.delta)
stWI.trim(starttime = stWI[0].stats.starttime+t00, endtime = stWI[0].stats.starttime+4+t00 )
stWI_Old = stWI.copy()

for trace in stWI:
        c = np.convolve(trace.data,y)
        c = c[0:len(trace)]
        trace.data = c

# Comb the signal a little bit
stWI.differentiate()

stWI.normalize()

stWI.sort()


kk=1.2
fE, axE = plt.subplots(1)    
fE.set_size_inches(15,5, forward=True)    

axE.plot(stN[1].times(),stN[1].data,'-b',30,linewidth=2)
axE.plot(stWI[0].times(),stWI[0].data+2 ,'--k',30,linewidth=2)
axE.text(3.9,kk-1+0.1,"Channel FD = " + "%s" % (stN[0].stats.channel), fontsize =12, color = "b")
axE.text(3.9,kk+0.1,"Channel WI = " + "%s" % (stWI[0].stats.channel), fontsize =12, color = "k")
axE.set_ylim(-2,4)

fN, axN = plt.subplots(1)    
fN.set_size_inches(15,5, forward=True)    

axN.plot(stN[0].times(),stN[0].data,'-b',30,linewidth=2)
axN.plot(stWI[1].times(),stWI[1].data+2 ,'--k',30,linewidth=2)
axN.text(3.9,kk-1+0.1,"Channel FD = " + "%s" % (stN[1].stats.channel), fontsize =12, color = "b")
axN.text(3.9,kk+0.1,"Channel WI = " + "%s" % (stWI[1].stats.channel), fontsize =12, color = "k")
axN.set_ylim(-2,4)

fZ, axZ = plt.subplots(1)    
fZ.set_size_inches(15,5, forward=True)    
# Flipped Z channel !!!!
axZ.plot(stN[2].times(),stN[2].data,'-b',30,linewidth=2)
axZ.plot(stWI[2].times(),-stWI[2].data+2 ,'--k',30,linewidth=2)
axZ.text(3.9,kk-1+0.1,"Channel FD = " + "%s" % (stN[2].stats.channel), fontsize =12, color = "b")
axZ.text(3.9,kk+0.1,"Channel WI = " + "%s" % (stWI[2].stats.channel), fontsize =12, color = "k")
axZ.set_ylim(-2,4)






    