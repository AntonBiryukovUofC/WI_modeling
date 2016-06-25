import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics,neighbors
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

Observations = np.genfromtxt("./Observations.csv",dtype = float,defaultfmt = "%.3e",delimiter = ',')
ClassLabels  = np.genfromtxt("./ClassLabels.csv",dtype = float,defaultfmt = "%d",delimiter = ',')

# Perturb the observations with the noise:
sigma = 0.009
mu=0
Observations = Observations + (sigma * np.random.randn(Observations.shape[0],Observations.shape[1]) + mu)
plt.plot(Observations[0,:])
X_train, X_test, y_train, y_test = train_test_split(Observations, ClassLabels, test_size=0.3, random_state=23)




#logreg = linear_model.LogisticRegressionCV(multi_class = "ovr",Cs= 20)
#EQfit = logreg.fit(X_train, y_train)
n_neighbors = 50
KNN =neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
EQfit = KNN.fit(X_train, y_train)



predicted = EQfit.predict(X_test)

probs = EQfit.predict_proba(X_test)
ScoreMetric =  metrics.accuracy_score(y_test, predicted)

# Analyze standard deviations per feature
stds = np.apply_along_axis(np.std, 0, Observations)
plt.plot(stds)

# Now when we learned on a low-res, lets apply to high res:
# Try to aggregate classes:
ObservationsDiffRes = np.genfromtxt("../8Class/Observations.csv",dtype = float,defaultfmt = "%.3e",delimiter = ',')
ClassLabelsDiffRes  = np.genfromtxt("../8Class/ClassLabels.csv",dtype = float,defaultfmt = "%d",delimiter = ',')

ObservationsDiffRes = ObservationsDiffRes + (sigma * np.random.randn(ObservationsDiffRes.shape[0],ObservationsDiffRes.shape[1]) + mu)
# delete non-midpoint classes:


OrigLab = ClassLabelsDiffRes

#ClassLabels8Class[np.in1d(ClassLabels8Class, (0,1))] = 0
#ClassLabels8Class[np.in1d(ClassLabels8Class, (2,3))] = 1
#ClassLabels8Class[np.in1d(ClassLabels8Class, (4,5))] = 2
#ClassLabels8Class[np.in1d(ClassLabels8Class, (6,7))] = 3

# Include midpoints only
inds_midpoints = np.in1d(ClassLabelsDiffRes, (1,2,5,6))
ClassLabelsDiffRes = ClassLabelsDiffRes[inds_midpoints]
# Drop the unnecessary observations:

ObservationsDiffRes = ObservationsDiffRes[inds_midpoints,:]


probsDiffRes= EQfit.predict_proba(ObservationsDiffRes)

PredictedDiffRes = EQfit.predict(ObservationsDiffRes)
#AccuracyDiffRes  = metrics.accuracy_score(ClassLabelsDiffRes, PredictedDiffRes)
#ConfMatrix  = metrics.confusion_matrix(ClassLabelsDiffRes,
 #                                    PredictedDiffRes)

for midPoint in [1,2,5,6]:
   midpoint_in_DiffRes = np.in1d(ClassLabelsDiffRes, (midPoint))
   averageValues = np.sum(probsDiffRes[midpoint_in_DiffRes,:],axis =0)
   #plt.plot(ObservationsDiffRes[midpoint_in_DiffRes,:][2])
   fig,ax = plt.subplots(1)
   ax.plot(averageValues)
   ax.set_title("Class %d " % (midPoint))
   



#scaler = preprocessing.StandardScaler(with_mean = False,with_std = False).fit(Observations)
