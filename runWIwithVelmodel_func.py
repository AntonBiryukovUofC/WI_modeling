import numpy as np
import commands
import matplotlib.pyplot as plt

from MiscFunctions import makeVelocityModel, MakeStationAndSourceFiles

def RunWIModel(prefix_dest= '', velname='VpVs.dat', fname = 'receiver.dat',sname = 'source.dat', tMax = 3.4,tensor = [1, 0, 0, 1, 0, 0]):
# Temporary: delete the station folders
#if len(glob.glob("station*"))>0:
#   map(shutil.rmtree,glob.glob("station*"))


# Build a velocity model file
    vel_filename = makeVelocityModel(velname)
# Build folders with stations
    station_names, stationCoords, stationAzimuths,sourceCoords = MakeStationAndSourceFiles(fname,sname,tMax,prefix_dest)
    stationCoords = np.matrix(stationCoords)

    plt.close('all')
    plt.plot(stationCoords[:,0],stationCoords[:,1],'ro')
    plt.plot(sourceCoords[0],sourceCoords[1],'go')
    plt.axis('equal')
    for i in range(len(stationAzimuths)):
        plt.annotate(str(i+1), (stationCoords[i,0],stationCoords[i,1]))
        plt.annotate(str(stationAzimuths[i]), (stationCoords[i,0]+100,stationCoords[i,1]+100))
        plt.savefig(prefix_dest + "receiver-source.png")
    i=-1
    #station_names.pop()

#tensor = [6.15, 10, 19.5, -34.3, -3.61, 34.1]
#returnnn
    print station_names
    for station_dest in station_names:
    # Prefix - for the folder where folders with station data are located
        station_dest = station_dest
        i+=1
        dfile = station_dest+"sta_dfile"
        FHS = station_dest + "eq_depth"
        FHR = station_dest + "sta_depth"
        command_to_hprep96 = "hprep96 -M %s -d %s -FHS %s -FHR %s -BH -ALL" % tuple ((vel_filename,dfile,FHS,FHR))  
        command_to_hspec96 = "hspec96 | tee hspec96.out"
        command_to_hpulse96 = "hpulse96 -D -p -l 1 > g1.vel"
        line_to_fmech96 = "fmech96 -XX %3.3f " + "-YY %3.3f "+ "-ZZ %3.3f " + "-XY %3.3f "+ "-XZ %3.3f "+ "-YZ %3.3f " + "-A %3.3f " + "-B %3.3f"  + "  < g1.vel | f96tosac -B" 
        if np.isnan(stationAzimuths[i]):
            stationAzimuths[i] = 0
        command_to_fmech96 = line_to_fmech96 % tuple((tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],stationAzimuths[i],stationAzimuths[i]+180))
        print station_dest
    

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



def RunWIModelNoTensor(prefix_dest= '', velname='VpVs.dat', fname = 'receiver.dat',sname = 'source.dat', tMax = 3.4):
# Temporary: delete the station folders
#if len(glob.glob("station*"))>0:
#   map(shutil.rmtree,glob.glob("station*"))


# Build a velocity model file
    vel_filename = makeVelocityModel(velname)
# Build folders with stations
    station_names, stationCoords, stationAzimuths,sourceCoords = MakeStationAndSourceFiles(fname,sname,tMax,prefix_dest)
    stationCoords = np.matrix(stationCoords)

    plt.close('all')
    plt.plot(stationCoords[:,0],stationCoords[:,1],'ro')
    plt.plot(sourceCoords[0],sourceCoords[1],'go')
    plt.annotate("source", (sourceCoords[0]+100,sourceCoords[1]+200))

    plt.axis('equal')
    for i in range(len(stationAzimuths)):
        plt.annotate(str(i+1), (stationCoords[i,0],stationCoords[i,1]))
        plt.annotate("%3.1f" % stationAzimuths[i], (stationCoords[i,0]+100,stationCoords[i,1]+250))
        plt.savefig(prefix_dest + "receiver-source.png")
    i=-1
    #station_names.pop()

#tensor = [6.15, 10, 19.5, -34.3, -3.61, 34.1]
#returnnn
    GVelFiles = list()
    for station_dest in station_names:
    # Prefix - for the folder where folders with station data are located
        station_dest = station_dest
        i+=1
        dfile = station_dest+"sta_dfile"
        FHS = station_dest + "eq_depth"
        FHR = station_dest + "sta_depth"
        outputStation = station_dest  + "gStation"+str(i+1)+".vel"
        GVelFiles.append(outputStation)
        command_to_hprep96 = "hprep96 -M %s -d %s -FHS %s -FHR %s -BH -ALL" % tuple ((vel_filename,dfile,FHS,FHR))  
        command_to_hspec96 = "hspec96 | tee hspec96.out"
        command_to_hpulse96 = "hpulse96 -D -p -l 1 > " + outputStation
       # line_to_fmech96 = "fmech96 -XX %3.3f " + "-YY %3.3f "+ "-ZZ %3.3f " + "-XY %3.3f "+ "-XZ %3.3f "+ "-YZ %3.3f " + "-A %3.3f " + "-B %3.3f"  + "  < g1.vel | f96tosac -B" 
        if np.isnan(stationAzimuths[i]):
            stationAzimuths[i] = 0
        # command_to_fmech96 = line_to_fmech96 % tuple((tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],stationAzimuths[i],stationAzimuths[i]+180))
        print station_dest
    

        status, output = commands.getstatusoutput(command_to_hprep96)
        print command_to_hprep96
        print output
    
        status, output = commands.getstatusoutput(command_to_hspec96)
        print command_to_hspec96
        #print output
    
        status, output = commands.getstatusoutput(command_to_hpulse96)
        print command_to_hpulse96
        print output
    
        #status, output = commands.getstatusoutput(command_to_fmech96)
        #print command_to_fmech96
        #print output
        #command_to_copy = "cp -v *.sac " + station_dest
        #status, output = commands.getstatusoutput(command_to_copy)   
        #print output
    with open(prefix_dest + "/" + "Azimuths", "w") as text_file:
            text_file.write(len(stationAzimuths)*"%3.2f " % tuple(stationAzimuths))
    return stationAzimuths,GVelFiles




    #continue
#    status, output = commands.getstatusoutput("rm *.{E,N,Z}0")
 #   status, output = commands.getstatusoutput("rm *.sac")
  #  status, output = commands.getstatusoutput("rm hspec96.{dat,grn,out}")
   # status, output = commands.getstatusoutput("rm g1.vel")

def ConvertGVelToSacWithTensor(prefix_dest = "./" ,tensor = [1, 0, 0, 1, 0, 0],stationAzimuths = [],gVelFiles = [], destination_copy = "./"): # Rotates the Green's functions into proper MT and saves the sac files
    for i in range(len(stationAzimuths)):
        line_to_fmech96 = "fmech96 -XX %3.3f " + "-YY %3.3f "+ "-ZZ %3.3f " + "-XY %3.3f "+ "-XZ %3.3f "+ "-YZ %3.3f " + "-A %3.3f " + "-B %3.3f"  + "  < %s | f96tosac -B" 
        
        command_to_fmech96 = line_to_fmech96 % tuple((tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5],stationAzimuths[i],stationAzimuths[i]+180,gVelFiles[i]))
        status, output = commands.getstatusoutput(command_to_fmech96)
        
        print command_to_fmech96
        stationID = "station%04d" % tuple([i+1])
        final_dest = destination_copy + "/" + stationID
        command_to_makedir = "mkdir -p " + final_dest
        status, output = commands.getstatusoutput(command_to_makedir)
        print command_to_makedir
        
        command_to_copy = "cp -v *.sac " + final_dest
        status, output = commands.getstatusoutput(command_to_copy)
        
        print command_to_copy
        
        with open(final_dest + "/" + "TensorComponents", "w") as text_file:
            text_file.write(len(tensor)*"%3.2f " % tuple(tensor))
        #print output

        

