import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

Observations = np.genfromtxt("./Observations.csv",dtype = float,defaultfmt = "%.3e",delimiter = ',')
ClassLabels  = np.genfromtxt("./ClassLabels.csv",dtype = float,defaultfmt = "%d",delimiter = ',')

# Perturb the observations with the noise:
sigma = 0.05
mu=0
Observations = Observations + (sigma * np.random.randn(Observations.shape[0],Observations.shape[1]) + mu)


plt.plot(Observations[0,:])
X_train, X_test, y_train, y_test = train_test_split(Observations, ClassLabels, test_size=0.3, random_state=23)

logreg = linear_model.LogisticRegressionCV(multi_class = "ovr")
EQfit = logreg.fit(X_train, y_train)



predicted = EQfit.predict(X_test)

probs = EQfit.predict_proba(X_test)
ScoreMetric =  metrics.accuracy_score(y_test, predicted)

# Analyze standard deviations per feature
stds = np.apply_along_axis(np.std, 0, Observations)
plt.plot(stds)

# Now when we learned on a low-res, lets apply to high res:
# Try to aggregate classes:
Observations8Class = np.genfromtxt("./8Class/Observations.csv",dtype = float,defaultfmt = "%.3e",delimiter = ',')
ClassLabels8Class  = np.genfromtxt("./8Class/ClassLabels.csv",dtype = float,defaultfmt = "%d",delimiter = ',')
# delete non-midpoint classes:

#raerasedr
OrigLab = ClassLabels8Class

ClassLabels8Class[np.in1d(ClassLabels8Class, (0,1))] = 0
ClassLabels8Class[np.in1d(ClassLabels8Class, (2,3))] = 1
ClassLabels8Class[np.in1d(ClassLabels8Class, (4,5))] = 2
ClassLabels8Class[np.in1d(ClassLabels8Class, (6,7))] = 3

# Predict on 10 class:
Predicted8Class = EQfit.predict(Observations8Class)
Accuracy8Class  = metrics.accuracy_score(ClassLabels8Class, Predicted8Class)
ConfMatrix  = metrics.accuracy_score(ClassLabels8Class,
                                     Predicted8Class)


#scaler = preprocessing.StandardScaler(with_mean = False,with_std = False).fit(Observations)
