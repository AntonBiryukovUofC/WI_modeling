# WI_modeling, Jan 2017
This repo is done for the purpose of implementing the modelling workflow based on waveform integration for earthquake waveform predictions and velocity calibration using MCMC approach.

Currently I am implementing the generalized Metropolis-Hastings MCMC to infer the velocity model and the elastic props of the layers in the toy example. Specifically, the codes in PythonMCMCForward.py are ray-tracing the travel times for P waves towards the stations from the earthquakes distributed withing some pre-defined volume.

The priority now is to implement Metropolis - Hastings Algorithm for the example and learn the sensitivity of the results to the station distribution, number of ground truth events involved , etc.


# General workflow for the modeling of the waveforms using freq-wavenumber integrator:
model96 -> hprep96 -> hspec96 -> hpulse96 -> file96

The program hprep96 creates a data file hspec96.dat for use by hspec96 to create the Green’s functions in the ω -distance space. The output of this program is a binary file, hspec96.grn, which is used by hpulse96 to convolve the response with the source time function to create file96 Green’s function time histories.

# Application of ML to the data

Specifically, I would like to work on improving the earthquake source location by combining stochastic and deterministic approaches.

My preliminary thought was to use the classifier, such as KNN to aid in constraining the volume in which the earthquake has potentially occurred. This classifier can be run on the waveform data from several seismic stations as an input, and then assign the xyz of the earthquake as a class.

How do I train the classifier? One way might be by running simulations with known source locations (XYZ or class) and form the training set from the synthetic waveforms. I can vary the moment tensor keeping the XYZ fixed of the event a little bit to achieve some variability in the features that would describe the same class.

Keeping the length of the waveforms fixed for all the simulations, I can concatenate the waveforms from several stations in one. For each sample of that long waveform I would thus be able to determine which station it belongs to and the time since the event origin it characterizes.
These samples therefore can form the feature set, and the signal amplitudes - the values of the features. 



# Miscellanious info for CPS codes



After Hpulse96 I need to specify Focal Mechanism Parameters.

# HPREP96
hprep96 [flags], where the command flags are

-M model 	Name of earth model file.

-d dfile	Name of distance file

-FHS srcdep  Name of source depth file

-FHR recdep  Name of receiver depth file

-TF (default true ) top surface is free

-BH (default true ) bottom surface is halfspace

-ALL (default true ) Compute all Green’s functions

-EQEX (default false) Compute earthquake/explosion Green’s functions

-EQF (default false) Compute explosion/point force Green’s functions

Creates a datafile hspec96.dat

#HSPEC96
The program requires the hspec96.dat file created by hprep96(VI). The program output is on stdout and on a binary file hspec96.grn.

hspec96 [flags], where the command flags are

## suggested usage: hspec96 > hspec.out

#HPULSE96
hpulse96
The program requires the hspec96.grn file created by hspec96(V) and optionally the source pulse definition file "rfile". The program output is on stdout and is a time series in file96(V) format. Program control is through the command line:

hpulse96 [flags], where the command flags are

-t Triangular pulse of base 2 L dt

-p Parabolic Pulse of base 4 L dt

-l L Source duration factor for the parabolic and triangular pulses.

-a alpha Shape parameter for Ohnaka pulse

-D Output is ground displacement

-V Output is ground velocity (default)

-A Output is ground acceleration

-F rfile User supplied pulse

-m mult Multiplier (default 1.0)

#FMECH96
fmech96 [flags], where the command flags are

-D dip dip of fault plane

-S Strike strike of fault plane

-R Rake slip angle on fault plane

-M0 Moment (def=1.0) Seismic moment in units of dyne-cm

-MW mw Moment magnitude

-E Explosion

-A Az Source to Station Azimuth ?

-B Baz (def=0) Station to Source azimuth ?

-ROT

Force the three component time histories to be vertical, radial and transverse
instead of vertical, north, and east. Since the Green’s functions are already vertical,
radial, and transverse, the value of the back-azimuth is not used.

-fx FX -fy Fy -fZ fz Point force amplitudes (N,E,down) in units of dynes 

-XX Mxx -YY Myy -ZZ Mzz -XY Mxy -XZ Mxz -YZ Myz  Moment tensor elements in units of dyne-cm

The relation between seismic moment, M 0 , and moment magnitude, M W used is
log 10 M 0 =1. 5M W +16. 05
