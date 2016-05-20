#!/bin/sh

### path to observed ####
DEST=/Files-ZHL/Inducedearthquake/sens_test/
####################################################
set -x
if [ ! -d SYNC_CLVD ]
then
	echo 'Creating the Greens function directory'
	mkdir SYNC_CLVD
fi
cd SYNC_CLVD
# do the wavenumber integration
for HS in 05
do
case $HS in
	0.5) DIRNAME=0005 ;;
	*) DIRNAME=0${HS}0 ;;
esac
if [ ! -d ${DIRNAME} ]
then
	echo 'creating the Greens Function depth direcotry'
	echo $DIRNAME 'for depth' $HS
	mkdir ${DIRNAME}
fi
cd $DIRNAME
# earth model

cat > AB_avr_vel4.mod << EOF
MODEL.01
AB AVRG4
ISOTROPIC
KGS
FLAT EARTH
1-D
CONSTANT VELOCITY
LINE08
LINE09
LINE10
LINE11
      H(KM)   VP(KM/S)   VS(KM/S) RHO(GM/CC)     QP         QS         ETAP   ETAS   FREFP  FREFS
     1.3000     2.6800     1.3500     2.3070   0.100E+03   0.600E+02   0.00   0.00   1.00   1.00
     1.3000     4.5900     2.3100     2.5557   0.300E+03   0.150E+03   0.00   0.00   1.00   1.00
     29.400     6.2000     3.5700     2.7041   0.300E+03   0.150E+03   0.00   0.00   1.00   1.00
     13.000     7.2000     4.1500     2.7651   0.300E+03   0.150E+03   0.00   0.00   1.00   1.00       
     0.0000     8.2000     4.7500     2.9733   0.100E+05   0.500E+04   0.00   0.00   1.00   1.00       
EOF
### making the elementary moment tensor synthetic seismograms
mkdir M
cp AB_avr_vel4.mod M/


for i in ${DEST}/data/*Z.SAC
do
DIST=`saclhdr -DIST $i`
AZIMUTH=`saclhdr -AZ $i`
BAZIMUTH=`saclhdr -BAZ $i`
cd M
cat > dfile << EOF
${DIST} 0.2 2048 -10 7.5
EOF
hprep96 -M AB_avr_vel4.mod -d dfile -HS ${HS} -HR 0.0 -EQEX
hspec96 > hspec96.out
hpulse96 -V -p -l 1 > g1.vel
DIP=60
STRIKE=30
RAKE=80
fmech96 -MW 4 -XX -0.82E+22 -YY 1.64E+22 -ZZ  -0.82E+22 -XY 0.00E+22 -XZ  0.00E+22 -YZ -0.00E+22 -A ${AZIMUTH} -B ${BAZIMUTH} -ROT  < g1.vel | f96tosac -G
rm g1.vel
rm hspec96*
rm dfile
cd ..
done
cd ..
done
cd ..
