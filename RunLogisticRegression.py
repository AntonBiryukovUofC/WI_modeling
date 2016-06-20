import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

Observations = np.genfromtxt("./Observations.csv",dtype = float,defaultfmt = "%.3e",delimiter = ',')
ClassLabels  = np.genfromtxt("./ClassLabels.csv",dtype = float,defaultfmt = "%d",delimiter = ',')

# Perturb the observations with the noise:
sigma = 1.0
mu=0
Observations = Observations + (sigma * np.random.randn(Observations.shape[0],Observations.shape[1]) + mu)


plt.plot(Observations[0,:])
X_train, X_test, y_train, y_test = train_test_split(Observations, ClassLabels, test_size=0.3, random_state=23)

logreg = linear_model.LogisticRegressionCV(Cs=10,multi_class = "ovr")
EQfit = logreg.fit(X_train, y_train)



predicted = EQfit.predict(X_test)

probs = EQfit.predict_proba(X_test)

# Analyze standard deviations per feature
stds = np.apply_along_axis(np.std, 0, Observations)
plt.plot(stds)




scaler = preprocessing.StandardScaler(with_mean = False,with_std = False).fit(Observations)
ScoreMetric =  metrics.accuracy_score(y_test, predicted)
score_eq = EQfit.score(X_test, y_test)