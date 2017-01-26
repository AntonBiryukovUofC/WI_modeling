import obspy
import utm
import glob
import numpy as np
import pandas as pd

#from plotly.plotly import iplot,plot
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.tools as tls 

def returnXYZ(x,y,z):
    return (x,y,z)

stationsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekStations.csv')
Xref = stationsCatalog.X[3]
Yref = stationsCatalog.Y[3]
stationsCatalog.X -=Xref
stationsCatalog.Y -=Yref
parent_dir = '/home/anton/NonLinLoc/foxcreek/loc'
list_locs = glob.glob(parent_dir+'/*.grid0.loc.hyp')
x=np.zeros(len(list_locs))
y=np.zeros(len(list_locs))
z=np.zeros(len(list_locs))
errZ=np.zeros(len(list_locs))
errHorMax=np.zeros(len(list_locs))
errHorMin=np.zeros(len(list_locs))

azHor = np.zeros(len(list_locs))
k=0
xrefUtm,yrefUtm = utm.from_latlon(stationsCatalog.Latitude[3],stationsCatalog.Longitude[3])[0:2]
i=0
for name in list_locs:
    try:
        cat = obspy.read_events(name,coordinate_converter = returnXYZ)
    except:
        print 'Could not read the event for some reason...'        
        k+=1
    #x[i],y[i] = utm.from_latlon(cat[0].origins[0].latitude,cat[0].origins[0].longitude)[0:2]
    Origin = cat[0].origins[0]
    x[i],y[i] = Origin.longitude,Origin.latitude
    z[i] = Origin.depth
    errZ[i] = Origin.depth_errors.uncertainty
    errHorMax[i] = Origin.origin_uncertainty.max_horizontal_uncertainty
    errHorMin[i] = Origin.origin_uncertainty.min_horizontal_uncertainty

    azHor[i] = Origin.origin_uncertainty.azimuth_max_horizontal_uncertainty
    print " Done %d out of %d " % (i,len(list_locs))
    i+=1
    
EventsDF = pd.DataFrame({'x':x,'y':y,'z':z,'errZ':errZ,'errHorMax':errHorMax,'errHorMin':errHorMin,'azHor':azHor})
#x-=xrefUtm
#y-=yrefUtm
z-=30
x*=1000
y*=1000


#norm1=mpl.colors.Normalize(vmin=0,vmax=2)
#m = cm.ScalarMappable(norm=norm1, cmap=cm.jet)

PltlyMapLayout = go.Layout(width=1200,height=1200,hovermode='closest')
PltlySrc = go.Scatter3d(y=y,x=x,z=-z,
                        mode = 'markers',
                       marker=dict(
                       size='10',
                       color = 'blue',
                       ),opacity=0.15
                       )       
PltlySta = go.Scatter3d(y=stationsCatalog.Y,x=stationsCatalog.X,z=stationsCatalog.X*0,
                        mode = 'markers',
                         error_x=dict(
                         type='percent',
                         value=10
                         ),
                         error_y=dict(
                         type='percent',
                         value=10
                         ),

                       marker=dict(
                       size='10',
                       color = 'black'                       
                       ))
                       
    

                       
data = [PltlySrc,PltlySta]
pltlyfig = go.Figure(data=data,layout = PltlyMapLayout)
plot(pltlyfig)
EventsDF.to_csv('NonLinLocEvents.csv')

