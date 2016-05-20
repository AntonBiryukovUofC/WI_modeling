# WI_modeling
This repo is done for the purpose of implementing the modelling workflow based on waveform integration for earthquake waveform predicitons
# General workflow for the modeling:
model96 -> hprep96 -> hspec96 -> hpulse96 -> file96

The program hprep96 creates a data file hspec96.dat
for use by hspec96 to create the Green’s functions in the
ω -distance space. The output of this program is a binary
file, hspec96.grn, which is used by hpulse96 to convolve
the response with the source time function to create file96
Green’s function time histories.


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
