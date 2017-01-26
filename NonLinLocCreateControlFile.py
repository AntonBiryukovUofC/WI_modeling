import numpy as np
import pandas as pd
from pyrocko import cake

with open("/home/anton/NonLinLoc/foxcreek/run/foxcreek_sample.in") as myfile:
        MainFileString="".join(line for line in myfile)
stationsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekStations.csv')

eventsCatalog = pd.read_csv('/home/anton/WI_Models/FoxCreekMSEED/foxCreekEvents.csv')
eventsCatalogSub = eventsCatalog[(eventsCatalog.Y<7000) & (eventsCatalog.Y>4000) & (eventsCatalog.X>3000) & (eventsCatalog.X<4500)]

Xref = stationsCatalog.X[3]
Yref = stationsCatalog.Y[3]
stationsCatalog.X -=Xref
stationsCatalog.Y -=Yref
stationsCatalog['Z'] = 30
MainFileString = MainFileString.replace('LATORIG','%3.2f' % stationsCatalog.Latitude[3])
MainFileString = MainFileString.replace('LONGORIG','%3.2f' % stationsCatalog.Longitude[3])
MainFileString = MainFileString.replace('ROTCW','%3.2f' % 0.0)

MainFileString = MainFileString.replace('OBSFORMAT','NLLOC_OBS')
MainFileString = MainFileString.replace('PATHTOOBS','/home/anton/NonLinLoc/foxcreek/obs/event*.obs')



dx=30
dy=30
dz=30
DepthMax= 5100.0
dN=30
NX = np.round(stationsCatalog.X.max()/dx)+dN
NY = np.round(stationsCatalog.Y.max()/dy)+dN
NZ = np.round(DepthMax/dz)

model =cake.load_model(('VpVs.nd'))
layer_model_string=''
for l in model.layers():
    layer_temp = 'LAYER   DEPTH  VPTOP 0.0    VSTOP  0.0  RHOTOP 0.0\n'
    layer_temp=layer_temp.replace('DEPTH','%3.2f' % (l.ztop/1000.0)    )
    layer_temp=layer_temp.replace('VPTOP','%3.2f' % (l.mtop.vp/1000.0)    )
    layer_temp=layer_temp.replace('VSTOP','%3.2f' % (l.mtop.vs/1000.0)    )
    layer_temp=layer_temp.replace('RHOTOP','%3.2f' % (l.mtop.rho/1000.0)    )
    layer_model_string +=layer_temp
layer_temp = 'LAYER   DEPTH  VPTOP 0.0    VSTOP  0.0  RHOTOP 0.0\n'

sta_string=''
for i,row in stationsCatalog.iterrows():
    sta_temp ='GTSRCE SRC_LABEL   XYZ  XSRC   YSRC   ZSRC   0.0\n'
    sta_temp =sta_temp.replace('SRC_LABEL',row.Name)    
    sta_temp =sta_temp.replace('XSRC','%3.3f' % (row.X/1000.0)    )
    sta_temp =sta_temp.replace('YSRC','%3.3f' % (row.Y/1000.0)    )
    sta_temp =sta_temp.replace('ZSRC','%3.3f' % (row.Z/1000.0)    )

    sta_string+=sta_temp
sta_temp ='GTSRCE SRC_LABEL   XYZ  XSRC   YSRC   ZSRC   0.0\n'

    
    


MainFileString = MainFileString.replace('NX','%d' % NX)
MainFileString = MainFileString.replace('NY','%d' % NY)
MainFileString = MainFileString.replace('NZ','%d' % NZ)

MainFileString = MainFileString.replace('N_X_LG','%d' %( NX-dN))
MainFileString = MainFileString.replace('N_Y_LG','%d' %( NY-dN))
MainFileString = MainFileString.replace('N_Z_LG','%d' %( NZ-dN))

MainFileString = MainFileString.replace('X0','%3.3f' % (-dN*dx/2000.0))
MainFileString = MainFileString.replace('Y0','%3.3f' % (-dN*dy/2000.0))
MainFileString = MainFileString.replace('Z0','%3.3f' % (0.0))

MainFileString = MainFileString.replace('X_0','%3.3f' % (-dx/1000.0))
MainFileString = MainFileString.replace('Y_0','%3.3f' % (-dy/1000.0))
MainFileString = MainFileString.replace('Z_0','%3.3f' % (0.0))


MainFileString = MainFileString.replace('DX','%3.3f' % (dx/1000.0))
MainFileString = MainFileString.replace('DY','%3.3f' % (dy/1000.0))
MainFileString = MainFileString.replace('DZ','%3.3f' % (dz/1000.0))
MainFileString = MainFileString.replace('MINNODESIZE','%3.3f' % (1.1*dz/1000.0))
MainFileString = MainFileString.replace('MAXNUMNODES','%d' % (NX*NY*NZ))
MainFileString = MainFileString.replace('LAYER   DEPTH  VPTOP VPGRAD    VSTOP  VSGRAD  RHOTOP RHOGRAD',layer_model_string)
MainFileString = MainFileString.replace('GTSRCE SRC_LABEL   XYZ  XSRC   YSRC   ZSRC   ELEVSRC',sta_string)
with open('/home/anton/NonLinLoc/foxcreek/run/foxcreek.in','w') as f:
    f.write(MainFileString)

