import numpy as np
import scipy.optimize as opt
from LocationsOnGrid import LocationsOnGridSmall

def costFunc(x,H,V,R):
    sum_term = H*V*x/np.sqrt(1-(x**2)*V**2);
    return R - sum(sum_term);

def GetHjVjRhoj(vels = None,rhos = None,depths=None, source_depth = -10):
        
    if source_depth < 0:
        print " Source depth is negative, halted.."
    
    ind_layers_above = np.argwhere(depths<source_depth)
    if ind_layers_above.shape[0]  == 0:
        print 'Event in the top layer'
        Hj = np.array([source_depth])
        Vj= np.array([vels[0]])
        rhoj= np.array([rhos[0]])
        return Hj,Vj, rhoj
        
        #returnn Hj,Vj
        
        
    Hj=np.hstack([depths[ind_layers_above].squeeze(), source_depth-depths[ind_layers_above[-1]]  ])
    Vj=vels[0:ind_layers_above.max()+2 ]
    rhoj = rhos[0:ind_layers_above.max()+2 ]
    if not(Vj.shape[0] == Hj.shape[0]):
        print 'The heights and vels are of different shape, go debug!'
    else:
        return Hj,Vj, rhoj
def CalculatePTime(vels = None,depths = None, 
                   rhos = None,
                   source_depth = 5000,
                   source_offset = 0) :
#vels =np.array([1550,3100, 6200])
#depths = np.array([2000, 4000])
#rhos = np.array([2.3,2.3, 2.7])

# Velocities for the segments v_j
# Thicknesses Hj
    R=source_offset
    H,V,Rho = GetHjVjRhoj(vels,rhos,depths,source_depth=3000)    
    p = opt.newton(func=costFunc,x0=1E-4,args=(H,V,R))
    
# create an array of cosines:
    cosV = np.sqrt(1-(p**2)*V**2);
# create an array of times per segment:
    t_int = H/(V*cosV);
    t_total = np.sum(t_int);
    return t_total