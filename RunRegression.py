import sklearn as sk
import numpy as np
# Import the observations:
f = "./Observations.csv"
Observations = np.genfromtxt(f,delimiter = ',',dtype = float)

f = "./ClassLabels.csv"
Labels = np.genfromtxt(f,delimiter = ',',dtype = int)

