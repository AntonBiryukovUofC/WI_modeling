import numpy as np
import scipy.optimize as opt
from LocationsOnGrid import LocationsOnGridSmall
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time
import numba
from joblib import Parallel, delayed

def GetHjVjRhoj(vels = [   3500,   3500,   3500,   3500,   3500,   3500],
                rhos = [2.32,2.55,2.75,2.32,2.55,2.75],
                depths=[ 2000,   3000,   4000,   5000,   6000],
                source_depth = 100):
        
    if source_depth < 0:
        print " Source depth is negative, halted.."
    
    ind_layers_above = np.argwhere(depths<source_depth)
    #print ind_layers_above
    h_diff=depths[ind_layers_above].squeeze()
    #print h_diff
    if ind_layers_above.shape[0]  == 0:
        print 'Event in the top layer'
        Hj = np.array([source_depth])
        Vj= np.array([vels[0]])
        rhoj= np.array([rhos[0]])
        return Hj,Vj, rhoj
        
        #returnn Hj,Vj
        
    elif np.size(h_diff) ==1 :
        Hj=np.hstack([h_diff,source_depth-h_diff  ])
        Vj=vels[0:ind_layers_above.max()+2 ]
        rhoj = rhos[0:ind_layers_above.max()+2 ]
        
        
    else:
        Hj=np.hstack([depths[0],np.diff(h_diff),
                  source_depth-depths[ind_layers_above[-1]]  ])
    
        Vj=vels[0:ind_layers_above.max()+2 ]
        rhoj = rhos[0:ind_layers_above.max()+2 ]
    if not(Vj.shape[0] == Hj.shape[0]):
        print 'The heights and vels are of different shape, go debug!'
        return Hj,Vj, rhoj

    else:
        return Hj,Vj, rhoj
def costFunc(x,H,V,R):
        sum_term = H*V*x/np.sqrt(1-(x**2)*V**2);
        return R - sum(sum_term);
    
def CalculatePTimePar(#vels = [   3500,   3500,   3500,   3500,   3500,   3500],
                   #depths = [ 2000,   3000,   4000,   5000,   6000], 
                   #rhos = [2.32,2.55,2.75,2.32,2.55,2.75],
                   #source_depth = 5000,
                   #source_offset = 0
                   x,costFunc=costFunc) :
    
    vels=x[2][0]
    depths=x[2][2]
    rhos=x[2][1]
    source_depth=x[0]
    source_offset=x[1]
#vels =np.array([1550,3100, 6200])
#depths = np.array([2000, 4000])
#rhos = np.array([2.3,2.3, 2.7])

# Velocities for the segments v_j
# Thicknesses Hj
    R=source_offset
    Hi,Vi,Rhoi = GetHjVjRhoj(vels,rhos,depths,source_depth) 
    
    res,r = opt.bisect(f=costFunc,a=0,b=1E-3,args=(Hi,Vi,R),full_output=True,disp=True)
    p=res
    #import pdb; pdb.set_trace()
# create an array of cosines:
    cosV = np.sqrt(1-(p**2)*Vi**2);
    #print Hi,Vi,cosV
# create an array of times per segment:
    t_int = Hi/(Vi*cosV);
    t_total = np.sum(t_int);
    return t_total



np.random.seed(seed=1)
                        
NXeq,NYeq,NZeq = 3,3,3
Neq = NXeq*NYeq*NZeq
print ' Setting EQ coordinates...'
xv,yv,zv,stationCoords =  LocationsOnGridSmall(receiver_name = 'receiver.dat',
                                               NX = NXeq, NY=NYeq,NZ=NZeq,leftBottomCorner=[200,200],
                                               rightTopCorner=[5000,5000],depthRange = [4300,4900])
x_perturb = np.random.uniform(low=150,high=700,size =xv.shape)
y_perturb = np.random.uniform(low=150,high=700,size =yv.shape)
z_perturb= np.random.uniform(low=100,high=200,size =zv.shape)
xv +=x_perturb
yv+=y_perturb
zv+=z_perturb
#returnn
# stCoords - create on random
print 'Setting the station coordinates...'
Nstations=5
x = np.random.uniform(low=200,high = 7000,size =Nstations)
y = np.random.uniform(low=200,high = 7000,size =Nstations)
z = 10*np.ones_like(x)
stCoords = np.stack((x,y,z),axis=1)
# Throw the coordinates of the stations and the earthquakes into corresponding DataFrames
stdf=pd.DataFrame({'x':x,'y':y,'z':z})
eqdf=pd.DataFrame({'x':xv.flatten(),'y':yv.flatten(),'z':zv.flatten()})

fig, ax = plt.subplots()
sns.regplot(x='x',y='y',data = stdf,fit_reg=False,scatter_kws = {'s':60},ax=ax)
sns.regplot(x='x',y='y',data = eqdf,fit_reg=False,scatter_kws = {'s':30},ax=ax,color='r')

# Number of iterations
vels = np.array([   3500,   3500,   3500,   3500,   3500,   3500])
rhos = np.array([2.32,2.55,2.75,2.32,2.55,2.75])
depths = np.array([ 2000,   3000,   4000,   5000,   6000])
N=40

t0 = time.time()

t0 = time.time()
input_list=[]
N=100
results=[]
input_list=[]
model = (vels,rhos,depths)
for eq_index,rowEq in eqdf.iterrows():
    eq_coords = np.array([rowEq.x,rowEq.y,rowEq.z])
    for st_index,rowSt in stdf.iterrows():
        eqCoords=rowEq.values
        source_depth = eqCoords[2]
        source_offset = np.sqrt((rowSt.x-eq_coords[0])**2 +(rowSt.y-eq_coords[1])**2)

        input_list.append([source_depth,source_offset,model])
for ii in range(N):
    
    
    
    inputs = input_list
    num_cores =6
    #returnn
    results = [CalculatePTimePar(i) for i in inputs]
    print ' Done with iter %d  ' % (ii)
    
        
        
        
t1 = time.time()
total = t1-t0  

print 'Total time is %3.6f s' % total   
     
'''
0.000 3.100 1.500 2.320
2.000 3.100 1.500 2.320
nd0
2.000 4.470 1.500 2.550
4.000 4.470 1.500 2.550
nd4
4.000 6.200 1.500 2.750
7.000 6.200 1.500 2.75
'''


