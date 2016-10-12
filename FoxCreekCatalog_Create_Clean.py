import pandas as pd
import numpy as np
import matplotlib as mpl
import os
import glob
from plotly.offline import download_plotlyjs, init_notebook_mode,  plot
import plotly.graph_objs  as go
import obspy
import utm
from MiscFunctions import GetPSArrivalRayTracing

def row_to_utm(row):
    #print row.Latitude, row.Longitude
    utmcoord = utm.from_latlon(row.Latitude, row.Longitude)
    return utmcoord[0],utmcoord[1]
    
def toUTC(row):
    import obspy
    date_list=[row.Year,row.Month,row.Day,row.Hour,row.Minute,row.Second]
    strUTC = obspy.UTCDateTime('%d-%02d-%02dT%02d:%02d:%2.3f' % tuple(date_list))
    return strUTC


events =pd.read_csv('./FullCatalog.csv')
stations= pd.read_csv('/home/anton/WI_Models/stalocs.csv',names =['Name','Latitude','Longitude','Elevation'])
stations.Longitude *=-1
events = events.dropna()
events = events[events['Depth (km)']<7]


events['ID']=range(events.shape[0])
events.index =events.ID
UTMEvents= np.array(events.apply(row_to_utm,axis=1))
UTMStations= np.array(stations.apply(row_to_utm,axis=1))
events[['X','Y']] = pd.DataFrame(UTMEvents,columns=['utm'])['utm'].apply(pd.Series)
stations[['X','Y']] = pd.DataFrame(UTMStations,columns=['utm'])['utm'].apply(pd.Series)
Xref = events['X'].mean()
Yref = events['Y'].mean()

events['X'] = events['X'] - Xref
events['Y'] = events['Y'] - Yref
events['UTCDate'] = events.apply(toUTC,axis=1)

stations['X'] = stations['X'] - Xref
stations['Y'] = stations['Y'] - Yref
PSta=np.zeros(events.shape[0])
SSta=np.zeros(events.shape[0])
for st_num in [1,2,3,4]:
    sta = stations.iloc[st_num-1]

    for index,row in events.iterrows():
        PSta[index],SSta[index],_,_ = GetPSArrivalRayTracing(sta_coords=np.array([sta.X,sta.Y,0]),
                                                         eq_coords=np.array([row.X,row.Y,row['Depth (km)']*1000]) )
                        
    events['PSta%d' % st_num] = PSta
    events['SSta%d' % st_num] = SSta
    events['dTSta%d' % st_num] = SSta - PSta




trace0 = go.Scatter(x=events.X,y=events.Y,
    mode='markers')

trace1 = go.Scatter(x=stations.X,y=stations.Y,
    mode='markers')
data = [trace0,trace1]
layout = go.Layout(
    showlegend=False,
    height=800,
    width=800,
    hovermode='closest',
)
fig = dict( data=data, layout=layout )
#plot(fig) 
parent_dir = '/home/anton/WI_Models/Foxcreek'
kp=0
kn=0
output_dir = './FoxCreekMSEED/'
#st_num = 2
#ch='HHZ'
for st_num in [1,2,3,4]:
    for ch in ['HHZ','HH1','HH2']:
        for index,row in events.iterrows():
            
                event_folder='%04d.%03d.%02d.%02d.*' % (row.Year,row.UTCDate.julday,row.Hour,row.Minute)
                stream_file = 'TE.WSK%02d.*%s*' % (st_num,ch)
                file_name = '%s/%s/%s' % (parent_dir,event_folder,stream_file)
                if len(glob.glob(file_name))<1:
                    print 'Cant find folder / stream in \n %s ' % file_name
                    kn+=1
                else:
                    st = obspy.read(file_name)
                    out_path = output_dir + '%04d/' % row.ID
                    if not(os.path.exists(out_path)):
                        os.makedirs(out_path)
                    print st
                    new_st_name = out_path+'WSK%02d%s.SAC' %(st_num,ch)
                    st.write(new_st_name,format='SAC')
                    kp+=1
events.to_csv(output_dir+'foxCreekEvents.csv')
    #    
    #    kn+=1

