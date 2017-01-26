from pyrocko import cake
import matplotlib
import numpy as np
sta_coords = np.array( [ 0,  0,    10.])
eq_coords =np.array([4000,6000,3910])
# play witht the Pyrocko modules

matplotlib.style.use('ggplot')
#from LocationsOnGrid import LocationsOnGridSmall
eq_depth = eq_coords[2]
so_offset = np.linalg.norm(sta_coords[:2] - eq_coords[:2])
so_offset = 7200

#so
#_,_,_,stCoords = LocationsOnGridSmall(receiver_name='receiver.dat',NX=1,NY = 1,NZ =1) # Get the receiver locations
#model =cake.load_model('VpVs.nd')
Distance = so_offset*cake.m2d
#model=cake.load_model('RepsolHighRes.nd')
p_transmission_paths = model.arrivals(distances = [Distance],phases = [cake.PhaseDef('p')],zstart = eq_depth,zstop=10)
s_transmission_paths = model.arrivals(distances = [Distance],phases = [cake.PhaseDef('s')],zstart = eq_depth,zstop=10)
for rayP,rayS in zip(p_transmission_paths,s_transmission_paths):
    p_arrival  = rayP.t
    print p_arrival
    s_arrival  = rayS.t
    print s_arrival
    
for rayP in p_transmission_paths:
    p_arrival  = rayP.t
    print p_arrival
    
for rayS in s_transmission_paths:
    s_arrival  = rayS.t
    print s_arrival