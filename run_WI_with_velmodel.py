import os
import numpy as np
# Start with the velocity model:
fname = 'VpVs.dat'
with open(fname) as f:
    layers = f.readlines()
layers.pop(0)
y=np.zeros((len(layers),len(str.split(layers[0]))))

i=0;
for line in layers:
    y[i,]=np.asarray(str.split(line)).astype(np.float)
    i+=1
if len(layers)>1:
    layer_thickness = np.hstack((y[0,0], y[1:y.shape[0],0]-y[0:y.shape[0]-1,0]))
y[:,0] = layer_thickness
    
    
vel_name = 'vel_model_model96'
with open(vel_name) as f:
    model_file = f.readlines()
template = model_file.pop()


