import shutil
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

from MiscFunctions import makeVelocityModel, MakeStationAndSourceFiles

# Temporary: delete the station folders
if len(glob.glob("station*"))>0:
    map(shutil.rmtree,glob.glob("station*"))


fname = 'VpVs.dat'
# Build a velocity model file
vel_filename = makeVelocityModel(fname)
fname = 'receiver.dat'
sname = 'source.dat'
# Build folders with stations
station_names, stationCoords, stationAzimuths = MakeStationAndSourceFiles(fname,sname)

stationCoords = np.matrix(stationCoords)
plt.plot(stationCoords[:,0],stationCoords[:,1],'ro')
plt.plot(stationCoords[-1,0],stationCoords[-1,1],'go')
plt.axis('equal')
for i in range(len(stationAzimuths)):
    plt.annotate(str(i), (stationCoords[i,0],stationCoords[i,1]))

for station_dest in station_names:
    #continue
    dfile = station_dest+"sta_dfile"
    FHS = station_dest + "eq_depth"
    FHR = station_dest + "sta_depth"
    command_to_hprep96 = "hprep96 -M %s -d %s -FHS %s -FHR %s -EQEX " % tuple ((vel_filename,dfile,FHS,FHR))  
    command_to_hspec96 = "hspec96 > hspec96.out"
    command_to_hpulse96 = "hpulse96 -V -p -l 1 > g1.vel"
    tensor_xx_yy_zz_xy_xz_yz = [1,1,1,0,0,0]
    command_to_fmech96 = "fmech96 -XX %3.3f -YY %3.3f -ZZ %3.3f -XY %3.3f -XZ %3.3f -YZ %3.3f -A %3.3f -ROT < g1.vel | f96tosac -G" # not done
    #rm g1.vel
    #rm hspec96*
    #rm dfile    
    
    print command_to_hprep96
    #os.system(command_to_hprep96)










