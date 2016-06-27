from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
import obspy.taup.taup_create as tp
#mymodel = build_taup_model("./ak135_copy.tvel",output_folder="./")
mymodel = build_taup_model("./TauPVpVs.tvel",output_folder="./")
#model = TauPyModel(model="./ak135_copy.npz")
model = TauPyModel(model="TauPVpVs.npz",verbose = True)
a = model.get_travel_times(source_depth_in_km=4,distance_in_degree=0.001,phase_list=["p","P","s","S"])
#b=model.get_ray_paths(source_depth_in_km=2.2,distance_in_degree=0.05,phase_list=["p","P","s","S","Pg","Pn","Sg","Sn","Ped","Sed"])
b=model.get_ray_paths(source_depth_in_km=2.2,distance_in_degree=0.05)
print(a)
