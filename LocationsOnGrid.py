import numpy as np
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

leftBottomCorner = np.asarray( [ min(stationCoords[:,0]) - 0.2*width, min(stationCoords[:,0]) - 0.2*height ] )
RightTopCorner = np.asarray( [ max(stationCoords[:,0])+0.2*width, max(stationCoords[:,0])+0.2*height ] )
