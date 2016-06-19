    
import numpy as np
    
def LocationsOnGrid(receiver_name = 'receiver.dat',NX = 20, NY=-2):
    # Get receiver locations:
    receiver_name = 'receiver.dat'
    with open(receiver_name) as f:
        stations = f.readlines()
        #stations.pop()
    stationCoords = list()
    for station in stations:
        if len(np.asarray(str.split(station)).astype(np.float)) == 0: continue
        coords = np.asarray(str.split(station)).astype(np.float)
        stationCoords.append(coords)
    stationCoords = np.matrix(stationCoords)
    width = max(stationCoords[:,0]) - min(stationCoords[:,0])
    height = max(stationCoords[:,1]) - min(stationCoords[:,1])
    # Get the coordinates for the search grid (find a smallest rectangle oriented in XY direction that fits the stations and the source)
    k=0.1
    leftBottomCorner = np.squeeze(np.asarray( [ np.min(stationCoords[:,0]) - k*width, np.min(stationCoords[:,0]) - k*height ] ))
    rightTopCorner =np.squeeze( np.asarray( [ np.max(stationCoords[:,0])+k*width, np.max(stationCoords[:,0])+k*height ] ))
    if NY<0:
        NY = np.floor((rightTopCorner[1] -leftBottomCorner[1])/(rightTopCorner[0] -leftBottomCorner[0])  * NX)
    
    x = np.linspace(leftBottomCorner[0],rightTopCorner[0],NX)
    y = np.linspace(leftBottomCorner[1],rightTopCorner[1],NY)
    xv, yv = np.meshgrid(x, y)
    return xv,yv