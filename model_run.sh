#----excecute with LAMMPI
#lamboot -v mpihosts
#lamboot -v
cd /home/anton/sw4/
./optimize/sw4  Model_Default.in | tee ./output.txt

