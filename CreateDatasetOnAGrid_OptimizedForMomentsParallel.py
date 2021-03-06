import numpy as np
from LocationsOnGrid import LocationsOnGrid
from runWIwithVelmodel_func import RunWIModelNoTensor,ConvertGVelToSacWithTensor
import subprocess

# 3166 4814 3910 - receiver at
# Get the location grid for the potential earthquake origins:

Ntensors = 400
nx_locations = 4

xv,yv = LocationsOnGrid(receiver_name='receiver.dat',NX=nx_locations,NY = 2)
sub_source_dir = "./SourcesGrid"
subprocess.call("rm -r "+ sub_source_dir, shell = True)

subprocess.call("mkdir "+ sub_source_dir, shell = True)

# 3166 4814 3910	0	2.5	2
for i in range(np.shape(xv)[0]):
    for j in range(np.shape(xv)[1]):
        with open(sub_source_dir + "/" + "source"+"_"+str(i)+"_"+str(j), "w") as text_file:
            text_file.write("%3.3f %3.3f %3.3f %3.3f %3.3f %3.3f" % (xv[i][j], yv[i][j], 3910,0,2.5,2))
            

# get the moment tensor
tensor = np.array([1,1,1,0,0,0])
sigma = 0.5/2
ClassLabel=0
for i in range(np.shape(xv)[0]):
    for j in range(np.shape(xv)[1]):
        location_no_tensor = "./"+"Class" +"%03d" % (ClassLabel) +"/"
        stationAzimuths,gVelFiles = RunWIModelNoTensor(
                                    prefix_dest= location_no_tensor,
                                    velname='VpVs.dat',fname = 'receiver.dat',
                                    sname = sub_source_dir + "/" + "source"+"_"+str(i)+"_"+str(j),
                                    tMax = 3.4
                                    )
        
        for k in range(Ntensors):
            tPerturb = np.random.normal(0,sigma,6)
            train_tensor = tensor + tPerturb
            #print train_tensor
            tensor_id_str = str(k)
            
            location_with_tensor = location_no_tensor + "moment" + tensor_id_str
            print location_with_tensor
            
            ConvertGVelToSacWithTensor(prefix_dest = '',tensor = train_tensor,
                                       stationAzimuths = stationAzimuths,
                                       gVelFiles = gVelFiles,
                                       destination_copy = location_with_tensor) 
                                       #save them in the same folder
        ClassLabel+=1
            
            
    

            
        
        #RunWIModel(prefix_dest= location_with_tensor, velname='VpVs.dat', fname = 'receiver.dat',sname = 'source.dat', tMax = 3.4,tensor = train_tensor)
