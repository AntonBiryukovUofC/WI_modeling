import shutil
import numpy as np
import obspy
import commands
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
tMax = 3.4

station_names, stationCoords, stationAzimuths = MakeStationAndSourceFiles(fname,sname,tMax)
stationCoords = np.matrix(stationCoords)
plt.plot(stationCoords[:,0],stationCoords[:,1],'ro')
plt.plot(stationCoords[-1,0],stationCoords[-1,1],'go')
plt.axis('equal')
for i in range(len(stationAzimuths)):
    plt.annotate(str(i), (stationCoords[i,0],stationCoords[i,1]))
    plt.annotate(str(stationAzimuths[i]), (stationCoords[i,0]+100,stationCoords[i,1]+100))

i=-1
station_names.pop()
Mxx = 6.15E20
Myy = 10E20
Mzz = 1.95E21
Mxy = -3.43E21 
Mxz = -3.61E20
Myz = 3.41E21

#tensor = [6.15, 10, 19.5, -34.3, -3.61, 34.1]
tensor = [1, 0, 0, 1, 0, 0]
#returnnn
for station_dest in station_names:
    #continue
    status, output = commands.getstatusoutput("rm *.{E,N,Z}0")
    status, output = commands.getstatusoutput("rm *.sac")

    status, output = commands.getstatusoutput("rm hspec96.{dat,grn,out}")
    status, output = commands.getstatusoutput("rm g1.vel")

    i+=1
    dfile = station_dest+"sta_dfile"
    FHS = station_dest + "eq_depth"
    FHR = station_dest + "sta_depth"
    command_to_hprep96 = "hprep96 -M %s -d %s -FHS %s -FHR %s -BH -ALL" % tuple ((vel_filename,dfile,FHS,FHR))  
    #command_to_hprep96 = "hprep96 -TH -M %s -d %s -FHS %s -FHR %s -EQEX " % tuple ((vel_filename,dfile,FHS,FHR))  
    command_to_hspec96 = "hspec96 | tee hspec96.out"
    #command_to_hpulse96 = "hpulse96 -D -i > g1.vel"
    command_to_hpulse96 = "hpulse96 -D -p -l 1 > g1.vel"
    #tensor_xx_yy_zz_xy_xz_yz = [1,1,1,0,0,0]
    #line_to_fmech96 = "fmech96 -XX %3.1f " + "-YY %3.1f "+ "-ZZ %3.1f " + "-XY %3.1f "+ "-XZ %3.1f "+ "-YZ %3.1f " + "-A %3.0f "+  " < g1.vel | f96tosac -B" 
    line_to_fmech96 = "fmech96 -XX %3.3f " + "-YY %3.3f "+ "-ZZ %3.3f " + "-XY %3.3f "+ "-XZ %3.3f "+ "-YZ %3.3f " + "-A %3.3f " + "-B %3.3f"  + "  < g1.vel | f96tosac -B" 
    if np.isnan(stationAzimuths[i]):
        stationAzimuths[i] = 0
    command_to_fmech96 = line_to_fmech96 % tuple((tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],stationAzimuths[i],stationAzimuths[i]+180))
    #command_to_fmech96 = line_to_fmech96 % tuple((tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],stationAzimuths[i]))
    
    #print command_to_fmech96
    print station_dest
    #rm g1.vel
    #rm hspec96*
    #rm dfile    
    
    
    status, output = commands.getstatusoutput(command_to_hprep96)
    print command_to_hprep96
    print output
    
    status, output = commands.getstatusoutput(command_to_hspec96)
    print command_to_hspec96
    #print output
    
    status, output = commands.getstatusoutput(command_to_hpulse96)
    print command_to_hpulse96
    print output
    
    status, output = commands.getstatusoutput(command_to_fmech96)
    print command_to_fmech96
    print output
    command_to_copy = "cp -v *.sac " + station_dest
    status, output = commands.getstatusoutput(command_to_copy)   
    print output







