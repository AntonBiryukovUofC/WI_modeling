import numpy as np
from LocationsOnGrid import LocationsOnGridSmall
from runWIwithVelmodel_func import RunWIModelNoTensor,ConvertGVelToSacWithTensor
import subprocess
import pandas as pd
import matplotlib as mpl
mpl.style.use('ggplot')


# 3166 4814 3910 - receiver at
# Get the location grid for the potential earthquake origins:

Ntensors = 50
nx_locations = 3
ny_locations = 3
nz_locations = 15
xv,yv,zv,stCoords = LocationsOnGridSmall(receiver_name='receiver.dat',NX=nx_locations,NY = ny_locations,NZ = nz_locations,
                                         leftBottomCorner=[3000,4000],rightTopCorner=[4500,6500],depthRange = [2500,5000])
sub_source_dir = "./SourcesGrid"
subprocess.call("rm -r "+ sub_source_dir, shell = True)

subprocess.call("mkdir "+ sub_source_dir, shell = True)
src_names = []
# 3166 4814 3910	0	2.5	2
for i in range(np.shape(xv)[0]):
    for j in range(np.shape(xv)[1]):
        for k in range(np.shape(xv)[2]):
            src_name  = sub_source_dir + "/" + "source"+"_"+str(i)+"_"+str(j)+"_"+str(k)
            src_names.append(src_name)
            with open(src_name, "w") as text_file:
                text_file.write("%3.3f %3.3f %3.3f %3.3f %3.3f %3.3f" % (xv[i][j][k], yv[i][j][k], zv[i][j][k],0,2.5,2))


sourcesDF =pd.DataFrame( {'X':xv.flatten(),'Y':yv.flatten(),'Z':zv.flatten(),'src_loc':src_names,'Class':range(len(src_names))} )
sourcesDF.index= range(sourcesDF.shape[0])
sourcesDF.to_csv('sourcesDF.csv')

Mxx = 6.15
Myy = 10
Mzz = 1.95E1
Mxy = -3.43E1 
Mxz = -3.61
Myz = 3.41E1

# get the moment tensor
tensor = np.array([Mxx,Myy,Mzz,Mxy,Mxz,Myz])
sigma = np.abs(tensor).mean()/6
for index,row in sourcesDF.iterrows():
        location_no_tensor = "./"+"Class" +"%03d" % (row.Class) +"/"
        stationAzimuths,gVelFiles = RunWIModelNoTensor(
                                    prefix_dest= location_no_tensor,
                                    velname='VpVs.dat',fname = 'receiver.dat',
                                    sname = row.src_loc,
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
            
            
    

            
        
        #RunWIModel(prefix_dest= location_with_tensor, velname='VpVs.dat', fname = 'receiver.dat',sname = 'source.dat', tMax = 3.4,tensor = train_tensor)
