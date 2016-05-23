import os
import shutil
import numpy as np
import glob
# Start with the velocity model:

def makeVelocityModel(filename):
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
    
fname = 'VpVs.dat'
#makeVelocityModel(fname)



receiver_name = "receiver.dat"
source_name = "source.dat"
n_per_2f = 30
tMax = 6.0

with open(source_name) as f:
    source = f.readlines()
    source_coords = np.asarray(str.split(source[0])).astype(np.float)
    dt = 1/source_coords[4]/n_per_2f
    n_of_two = np.round(np.log(tMax/dt)/np.log(2))


i=0
with open(receiver_name) as f:
    stations = f.readlines()
    
for station in stations:
    coords = np.asarray(str.split(station)).astype(np.float)
    if len(coords) == 0: continue
    i+=1;
    path_to_station = "./station" + ("%04d" % i)
    os.makedirs(path_to_station)
    # Save depth into a file
    with open(path_to_station + "/" + "sta_depth", "w") as text_file:
        text_file.write("%3.3f" % (float(coords[2])/1000))
    















