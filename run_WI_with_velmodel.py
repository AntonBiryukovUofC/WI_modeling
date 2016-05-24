import os
import shutil
import numpy as np
import glob
import matplotlib.pyplot as plt
# Coordinate conversion for azimuths:
def DistAndAz(x, y):
    rho = np.sqrt(x**2 + y**2)
    #phi = np.arctan2(y, x)
    if x>0:
        phi = 90-np.arctan(y/x)/(2*np.arccos(0))*180   
        #print phi
    else:
        phi = 270 -np.arctan(y/x)/(2*np.arccos(0))*180
    
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# Start with the velocity model:

def makeVelocityModel(filename):
    fname = filename
    with open(fname) as f:
        layers = f.readlines()
    layers.pop(0)
    y=np.zeros((len(layers),4+len(str.split(layers[0]))))
    
    
    vel_name = 'vel_model_model96'
    with open(vel_name) as f:
        model_file = f.readlines()
    model_file.pop()
    
    i=0;
    for line in layers:
        y[i,]=np.hstack((np.asarray(str.split(line)).astype(np.float),[0,0,1,1]))
        y[i,1:4] = y[i,1:4]/1000
        i+=1
    if len(layers)>1:
        layer_thickness = np.hstack((y[0,0], y[1:y.shape[0],0]-y[0:y.shape[0]-1,0]))/1000
        
    y[:,0] = layer_thickness
    vel_to_add = str("")
    for i in range(y.shape[0]):
        vel_to_add = vel_to_add + (" %f"*10)[1:] % tuple(y[i,:]) + "\n"
        
    final_vel = "".join(model_file) + vel_to_add
    
    with open("Vel_Model_Final", "w") as text_file:
        text_file.write(" %s" % final_vel)
    return
# Temporary: delete the station folders
if len(glob.glob("station*"))>0:
    map(shutil.rmtree,glob.glob("station*"))
    

def MakeStationAndSourceFiles(Rec_filename,Source_filename):
    #receiver_name = "receiver.dat"
    receiver_name = Rec_filename
    #source_name = "source.dat"
    source_name = Source_filename
    n_per_2f = 40
    tMax = 6.0
    
    with open(source_name) as f:
        source = f.readlines()
        source_coords = np.asarray(str.split(source[0])).astype(np.float)
        dt = 0.5/source_coords[4]/n_per_2f
        n_of_two = np.float(np.round(np.log(tMax/dt)/np.log(2)))
    nPts = np.float(2)**n_of_two
    
    i=0
    with open(receiver_name) as f:
        stations = f.readlines()
    stations.pop()
    #stations.pop()
    station_list = list()    
    sta_azimuths = list()
    stationCoords = list()

    for station in stations:
        if len(np.asarray(str.split(station)).astype(np.float)) == 0: continue
        coords = np.asarray(str.split(station)).astype(np.float)
        stationCoords.append(coords)
        i+=1;
        path_to_station = "./station" + ("%04d" % i)
        os.makedirs(path_to_station)
        # Save depth into a file
        with open(path_to_station + "/" + "sta_depth", "w") as text_file:
            text_file.write("%3.3f" % (float(coords[2])/1000))
            station_list.append(path_to_station + "/" + "sta_depth")
        with open(path_to_station + "/" + "eq_depth", "w") as text_file:
            text_file.write("%3.3f" % (float(source_coords[2])/1000))
        # Save a distance file
        #    DIST DT NPTS T0 VRED
        rec_m_s = (-source_coords[0:2] + coords[0:2])
        dist, azimuth = DistAndAz(rec_m_s[0],rec_m_s[1])
        dist = dist/1000 # to km
        #azimuth =azimuth/2*np.arccos(0) * 180
        sta_azimuths.append(azimuth)
        with open(path_to_station + "/" + "sta_dfile", "w") as text_file:
            text_file.write("%3.3f %3.5f %d %3.3f %3.3f" % tuple((dist,dt,nPts,0,0)))
    return station_list, stationCoords, sta_azimuths

fname = 'VpVs.dat'
# Build a velocity model file
makeVelocityModel(fname)
fname = 'receiver.dat'
sname = 'source.dat'
# Build folders with stations
station_names, stationCoords, stationAzimuths = MakeStationAndSourceFiles(fname,sname)
stationCoords = np.matrix(stationCoords)
plt.plot(stationCoords[:,0],stationCoords[:,1],'ro')
plt.plot(stationCoords[3,0],stationCoords[3,1],'go')
plt.axis('equal')

for i in range(len(stationAzimuths)):
    plt.annotate(str(i), (stationCoords[i,0],stationCoords[i,1]))












