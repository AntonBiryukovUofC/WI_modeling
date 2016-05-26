import obspy
import glob
# Get the traces from SW4


root_dir = "/home/anton/Matlab_Data/Model_Default/"
channel="y"
list_sac =  glob.glob(root_dir +'*'+ "." + channel);
list_sac.sort();
stN=obspy.Stream();
stWI=obspy.Stream();

for file in list_sac:
    stN+=obspy.read(file)
#stN.filter("bandpass",freqmin = 0.0005,freqmax = 15)
stN[1].plot(type= 'relative')
num_sta = "2"
list_WI_sac =  glob.glob("/home/anton/WI_Models/station000"+num_sta+"/*.sac")
#list_WI_sac =  glob.glob("/home/anton/WI_Models/"+"*.sac")
for file in list_WI_sac:
    stWI+=obspy.read(file)
#stN.filter("bandpass",freqmin = 0.0005,freqmax = 5)
#stWI.integrate()
#stWI.filter("bandpass",freqmin = 0.0005,freqmax = 8)

stWI.normalize()
stWI.trim(starttime = stWI[0].stats.starttime, endtime = stWI[0].stats.starttime+4 )
stWI.plot(type= 'relative')




    