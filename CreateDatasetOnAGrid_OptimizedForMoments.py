import numpy as np
from LocationsOnGrid import LocationsOnGrid
from runWIwithVelmodel_func import RunWIModelNoTensor
import subprocess

# 3166 4814 3910 - receiver at
# Get the location grid for the potential earthquake origins:

Ntensors = 8
nx_locations = 10

xv,yv = LocationsOnGrid(receiver_name='receiver.dat',NX=nx_locations)
sub_source_dir = "./SourcesGrid"
subprocess.call("rm -r "+ sub_source_dir, shell = True)

subprocess.call("mkdir "+ sub_source_dir, shell = True)

# 3166 4814 3910	0	2.5	2
for i in range(np.shape(xv)[0]):
    for j in range(np.shape(xv)[1]):
        with open(sub_source_dir + "/" + "source"+"_"+str(i)+"_"+str(j), "w") as text_file:
            text_file.write("%3.3f %3.3f %3.3f %3.3f %3.3f %3.3f" % (xv[i][j], yv[i][j], 3910,0,2.5,2))
            

# get the moment tensor
tensor = np.array([0,0,0,1,0,1])
sigma = 0.5/2
for i in range(np.shape(xv)[0]):
    for j in range(np.shape(xv)[1]):
        location_no_tensor = "./"+"Row"+str(i)+"Col"+str(j) + "/"
        RunWIModelNoTensor(prefix_dest= './temp/', velname='VpVs.dat', fname = 'receiver.dat',sname = sub_source_dir + "/" + "source"+"_"+str(i)+"_"+str(j),
                           tMax = 3.4)
        
        for k in range(Ntensors):
            tPerturb = np.random.normal(0,sigma,6)
            train_tensor = tensor + tPerturb
            #print train_tensor
            
            tensor_id_str = str(k)
            
            print location_with_tensor
            
            
    

            
        
        #RunWIModel(prefix_dest= location_with_tensor, velname='VpVs.dat', fname = 'receiver.dat',sname = 'source.dat', tMax = 3.4,tensor = train_tensor)
