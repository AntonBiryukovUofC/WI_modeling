import obspy

# Get the traces from SW4


root_dir = "/home/anton/Matlab_Data/Model_Default/"
channel="z"
list_sac =  glob.glob(root_dir +'*'+ "." + channel);
list_sac.sort();
stN=obspy.Stream();
for file in list_sac:
    stN+=obspy.read(file)
    
    