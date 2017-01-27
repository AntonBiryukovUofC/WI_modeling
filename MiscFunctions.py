import os
import numpy as np
import obspy

# Ricker wavelet for convolution
def ricker(f, length=0.512, dt=0.001):
    t = np.linspace(-length/2, (length-dt)/2, length/dt)
    y = (1.-2.*(np.pi**2)*(f**2)*(t**2))*np.exp(-(np.pi**2)*(f**2)*(t**2))
    return t, y

def rickerInt(t0,tmin,tmax,f, dt=0.001):  
    omega = f
    n = round((tmax-tmin)/dt)
    t = tmin + np.linspace(0,n,n+1)*dt
    #print t
    g = (t-t0)*np.exp(-(np.pi*omega*(t-t0))**2)
    return t,g
def VerySmoothBump(t0,tmin,tmax,f, dt=0.001):
    n = int(round((tmax-tmin)/dt))
    t = tmin + np.linspace(0,n,n+1)*dt
    omega=f
    y=t*0
    print n
    for i in range(n+1):
        ta = omega*(t[i]-t0)
        if ta<0:
            y[i] = 0
        else:
            if ta>1:
                y[i] = 0
            else:
                y[i] = - 1024*ta**10 + 5120*ta**9 - 10240*ta**8 + 10240*ta**7 - 5120*ta**6 + 1024*ta**5
    return t,y
    
   
# Coordinate conversion for azimuths:
def DistAndAz(x, y):
    rho = np.sqrt(x**2 + y**2)
    #phi = np.arctan2(y, x)
    if x>0:
        phi = 90-np.arctan(y/x)/(2*np.arccos(0))*180   
        #print phi
    else:
        phi = 270 -np.arctan(y/x)/(2*np.arccos(0))*180
    
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

# Start with the velocity model:

def makeVelocityModel(filename):
    fname = filename
    with open(fname) as f:
        layers = f.readlines()
    layers.pop(0)
    y=np.zeros((len(layers),4+len(str.split(layers[0]))))
    
    
    vel_name = 'vel_model_model96'
    with open(vel_name) as f:
        model_file = f.readlines()
    model_file.pop()
    
    i=0;
    for line in layers:
        y[i,]=np.hstack((np.asarray(str.split(line)).astype(np.float),[0,0,1,1]))
        y[i,1:4] = y[i,1:4]/1000
        i+=1
    if len(layers)>1:
        layer_thickness = np.hstack((y[0,0], y[1:y.shape[0],0]-y[0:y.shape[0]-1,0]))/1000
        
    y[:,0] = layer_thickness
    vel_to_add = str("")
    for i in range(y.shape[0]):
        vel_to_add = vel_to_add + (" %f"*10)[1:] % tuple(y[i,:]) + "\n"
        
    final_vel = "".join(model_file) + vel_to_add
    
    with open("Vel_Model_Final", "w") as text_file:
        text_file.write("%s" % final_vel)
    final_name = "Vel_Model_Final"
    return final_name

    
# Make stations and receivers files
def MakeStationAndSourceFiles(Rec_filename,Source_filename,tMax,prefix_dest = ''  ):
    #receiver_name = "receiver.dat"
    receiver_name = Rec_filename
    #source_name = "source.dat"
    source_name = Source_filename
    n_per_2f = 10
    
    with open(source_name) as f:
        source = f.readlines()
        source_coords = np.asarray(str.split(source[0])).astype(np.float)
        dt = 0.5/source_coords[4]/n_per_2f
        n_of_two = np.float(np.floor(np.log(tMax/dt)/np.log(2))+1)
    #print tMax/dt    
    nPts = np.float(2)**n_of_two
    #print nPts
    
    i=0
    with open(receiver_name) as f:
        stations = f.readlines()
    stations.pop()
    #stations.pop()
    station_list = list()    
    sta_azimuths = list()
    stationCoords = list()

    for station in stations:
        if len(np.asarray(str.split(station)).astype(np.float)) == 0: continue
        coords = np.asarray(str.split(station)).astype(np.float)
        stationCoords.append(coords)
        i+=1;
        path_to_station =  prefix_dest + "station" + ("%04d" % i)
        os.makedirs(path_to_station)
        # Save depth into a file
        with open(path_to_station + "/" + "sta_depth", "w") as text_file:
            text_file.write("%3.3f" % (float(coords[2])/1000))
            station_list.append(path_to_station + "/")
        with open(path_to_station + "/" + "eq_depth", "w") as text_file:
     #       print source_coords
            text_file.write("%3.3f" % (float(source_coords[2])/1000))
        # Save a distance file
        #    DIST DT NPTS T0 VRED
        rec_m_s = (-source_coords[0:2] + coords[0:2])
        dist, azimuth = DistAndAz(rec_m_s[0],rec_m_s[1])
        dist = dist/1000 # to km
        #azimuth =azimuth/2*np.arccos(0) * 180
        sta_azimuths.append(azimuth)
        with open(path_to_station + "/" + "sta_dfile", "w") as text_file:
            text_file.write("%3.3f %3.5f %d %3.3f %3.3f" % tuple((dist,dt,nPts,0,0)))
    return station_list, stationCoords, sta_azimuths, source_coords


def circshift(tr, ind):
    """
    circular shift of tr by ind samples
    USAGE
    trshift = circshift(tr, ind)
    INPUTS
    tr - trace to shift
    ind - number of samples to shift tr.data
    OUTPUTS
    """
    trshift = tr[0].copy()
    trshift.data = np.roll(trshift.data, ind)
    #trshift.stats.starttime = trshift.stats.starttime + ind*(1./trshift.stats.sampling_rate)
    trshift = obspy.Stream(trshift)
    return trshift

def GetPSArrivalRayTracing(sta_coords = np.array([0,0,0.0]), eq_coords =np.array([0,0,3900]),model_name = 'VpVs.nd'):
    # play witht the Pyrocko modules
    from pyrocko import cake
    import matplotlib
    matplotlib.style.use('ggplot')
    from LocationsOnGrid import LocationsOnGridSmall
    eq_depth = eq_coords[2]
    so_offset = np.linalg.norm(sta_coords[:2] - eq_coords[:2])
    model =cake.load_model('VpVs.nd')
    _,_,_,stCoords = LocationsOnGridSmall(receiver_name='receiver.dat',NX=1,NY = 1,NZ =1) # Get the receiver locations
    
    Distance = so_offset*cake.m2d
    p_transmission_paths = model.arrivals(distances = [Distance],phases = [cake.PhaseDef('p')],zstart = eq_depth)
    s_transmission_paths = model.arrivals(distances = [Distance],phases = [cake.PhaseDef('s')],zstart = eq_depth)
    for rayP,rayS in zip(p_transmission_paths,s_transmission_paths):
        p_arrival  = rayP.t
        print p_arrival
        s_arrival  = rayS.t
        print s_arrival
    return p_arrival,s_arrival,so_offset,model

def GetPSArrivalRayTracingMC(sta_coords = np.array([0,0,0.0]), eq_coords =np.array([0,0,3900]),model=None,mode='POnly'):
    # play witht the Pyrocko modules
    from pyrocko import cake
 
    #from LocationsOnGrid import LocationsOnGridSmall
    eq_depth = eq_coords[2]
    so_offset = np.linalg.norm(sta_coords[:2] - eq_coords[:2])
    #_,_,_,stCoords = LocationsOnGridSmall(receiver_name='receiver.dat',NX=1,NY = 1,NZ =1) # Get the receiver locations
    
    Distance = so_offset*cake.m2d
    
    p_transmission_paths = model.arrivals(distances = [Distance],phases = [cake.PhaseDef('p')],zstart = eq_depth,zstop=sta_coords[2])
    for rayP in p_transmission_paths:
        p_arrival  = rayP.t
        #print p_arrival
    if len(p_transmission_paths)<1:
        p_arrival = None
        
    if not(mode == 'POnly'):
        s_transmission_paths = model.arrivals(distances = [Distance],phases =   [cake.PhaseDef('s')],
                                              zstart = eq_depth,zstop=sta_coords[2])
        for rayS in s_transmission_paths:
            s_arrival  = rayS.t
            #print s_arrival
    else:
            s_transmission_paths=[]
            
    if len(s_transmission_paths)<1:
        s_arrival = None
        
        
    return p_arrival,s_arrival,so_offset


def getAmplitudeEnvelopeFeatures(traceName = '/home/anton/WI_Models/AllTraces/M0055_station_0003_location_Class036_channel_Z.mseed',st=1.2,fn=2.2):
    import obspy 
    import numpy as np
    from obspy.signal.filter import envelope
    from scipy.integrate import simps
    from scipy.stats import kurtosis
    trace = obspy.read(traceName)
    trace.normalize()
    #data_envelopeM = obspy.signal.filter.envelope(trace.data)
    TraceCopy  = trace[0].copy()
    envTrace = trace[0].copy()
   
    envTrace.data = envelope(TraceCopy.data)
    # Feature 0 : peakedness 
    KurtosisEnvelopeDiff =kurtosis(np.diff(envTrace.data))
    # Feature 1
    StdEnvelope = envTrace.data.std()
    # Feature 2
    MeanEnvelope = envTrace.data.mean()
    
    
    envTrace.trim(starttime = trace[0].stats.starttime + st,endtime = trace[0].stats.starttime + fn)
    TraceCopy.trim(starttime = trace[0].stats.starttime + st,endtime = trace[0].stats.starttime + fn)
    #Integrate the envelope in the region:
    # Feature 3
    EnvelopeIntegral = simps(y = envTrace.data,x = envTrace.times()) / (envTrace.stats.endtime - envTrace.stats.starttime)
    EnergyInTheSubTrace = simps(y = TraceCopy.data**2,x = TraceCopy.times())
    EnergyWholeTrace = simps(y = trace[0].data**2,x = trace[0].times())
    
    #Feature 4
    EnergyRatio = EnergyInTheSubTrace/EnergyWholeTrace
    #Feature 5 
    zc= np.sign( np.diff(envTrace.data) )  
    zc[zc==0] = -1     # replace zeros with -1  
    zcFeature = np.where(np.diff(zc))[0].shape[0] 
    
    return KurtosisEnvelopeDiff,StdEnvelope,MeanEnvelope,EnvelopeIntegral,EnergyRatio,zcFeature

def getAmplitudeEnvelopeFeaturesReal(traceName = '/home/anton/WI_Models/AllTraces/M0055_station_0003_location_Class036_channel_Z.mseed',
                                     st=1.2,fn=2.2,fmin=1,fmax=10,starttime=None,endtime=None ):
    import obspy 
    import numpy as np
    from obspy.signal.filter import envelope
    from scipy.integrate import simps
    from scipy.stats import kurtosis
    trace = obspy.read(traceName)
    if trace[0].stats.starttime.year < starttime.year:
        return None
    trace.taper(type= "hann",max_percentage=0.2)    

    trace.filter(type='bandpass',freqmin=fmin,freqmax=fmax)
    trace.trim(starttime = starttime,endtime = endtime)
    trace.normalize()
    trace.taper(type= "cosine",max_percentage=0.05)    
    trace.plot(type='relative')
    #data_envelopeM = obspy.signal.filter.envelope(trace.data)
    TraceCopy  = trace[0].copy()
    envTrace = trace[0].copy()
   
    envTrace.data = envelope(TraceCopy.data)
    # Feature 0 : peakedness 
    KurtosisEnvelopeDiff =kurtosis(np.diff(envTrace.data))
    # Feature 1
    StdEnvelope = envTrace.data.std()
    # Feature 2
    MeanEnvelope = envTrace.data.mean()
    
    
    envTrace.trim(starttime = trace[0].stats.starttime + st,endtime = trace[0].stats.starttime + fn)
    TraceCopy.trim(starttime = trace[0].stats.starttime + st,endtime = trace[0].stats.starttime + fn)
    #Integrate the envelope in the region:
    # Feature 3
    EnvelopeIntegral = simps(y = envTrace.data,x = envTrace.times()) / (envTrace.stats.endtime - envTrace.stats.starttime)
    EnergyInTheSubTrace = simps(y = TraceCopy.data**2,x = TraceCopy.times())
    EnergyWholeTrace = simps(y = trace[0].data**2,x = trace[0].times())
    
    #Feature 4
    EnergyRatio = EnergyInTheSubTrace/EnergyWholeTrace
    #Feature 5 
    zc= np.sign( np.diff(envTrace.data) )  
    zc[zc==0] = -1     # replace zeros with -1  
    zcFeature = np.where(np.diff(zc))[0].shape[0] 
    
    return KurtosisEnvelopeDiff,StdEnvelope,MeanEnvelope,EnvelopeIntegral,EnergyRatio,zcFeature

def getNonLinLocPhaseLine(de =obspy.UTCDateTime('2015-01-04T07:10:50.047000Z'),sta='WSK01',ch='Z',phase = 'P'):
    # Get the Phase File:
    string_phase="%6s %4s %4s %1s %6s %1s %s GAU %9.2e %9.2e %9.2e %9.2e\n"
    
    instr = 'BB'
    
    onset='?'
    firstMotion='?'
    errmag=0.0
    coda=-1.0
    amp=-1.0
    period=-1.0
    date = '%4d%02d%02d %02d%02d %7.4f' % (de.year,de.month,de.day,de.hour,de.minute,(de.second+de.microsecond*1e-6))
    phase_complete = string_phase % (sta,instr,ch,onset,phase,firstMotion,date,errmag,coda,amp,period)
    return phase_complete
    

def row_to_utm(row):
    import utm

    #print row.Latitude, row.Longitude
    utmcoord = utm.from_latlon(row.Latitude, row.Longitude)
    return utmcoord[0],utmcoord[1]
    
def toUTC(row):
    import obspy
    date_list=[row.Year,row.Month,row.Day,row.Hour,row.Minute,row.Second]
    strUTC = obspy.UTCDateTime('%d-%02d-%02dT%02d:%02d:%2.3f' % tuple(date_list))
    return strUTC

def DoForwardModel(eqdf,stdf,model):
    Neq=eqdf.shape[0]
    Nstations = stdf.shape[0]
    tp=np.zeros((Neq,Nstations))
    ts=np.zeros_like(tp)
    so=np.zeros_like(tp)
    for eq_index,rowEq in eqdf.iterrows():
        eq_coords = np.array([rowEq.x,rowEq.y,rowEq.z])
        for st_index,rowSt in stdf.iterrows():
            # Get the time for P,S arrivals and offset
            p,s,o= GetPSArrivalRayTracingMC(sta_coords=[rowSt.x,rowSt.y,rowSt.z],
                                               eq_coords=eq_coords,
                                               model=model)
            tp[eq_index,st_index],ts[eq_index,st_index],so[eq_index,st_index]=p,s,o 
            
            print ' Done with station %d and eq %d ' % (st_index,eq_index)
    return tp,ts,so
