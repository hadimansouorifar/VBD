import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras import optimizers
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.optimizers import SGD,Adam
import numpy
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve
from imblearn.under_sampling import RandomUnderSampler
import random

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

from sklearn.covariance import EmpiricalCovariance, MinCovDet

OPTIMIZER = Adam(lr=0.0001, decay=8e-9)
from sklearn import svm



# Define model
model = Sequential()
model.add(Dense(18, input_dim=23, activation= "relu"))
model.add(Dense(12, activation= "relu"))
model.add(Dense(6, activation= "relu"))
model.add(Dense(12, activation= "relu"))
model.add(Dense(18, activation= "relu"))
model.add(Dense(23, activation= "sigmoid"))

   



filename = 'parkinson.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
td = list(reader)
data = numpy.array(td).astype('str')



x = data[:, 1:23]  # select columns 1 through end


x = numpy.array(x).astype('float')

w = data[:, 23]   # select column 0, the stock priceprint(w)
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)
w = numpy.array(w).astype('float')




for i in range(0, len(x)):

    if (w[i] == 0):
        w[i]=1

    else:
        w[i] = 0



amount_scaler = StandardScaler().fit(x[:])
x[:] = amount_scaler.transform(x[:])

print(len(x))

model.compile(loss= "mean_squared_error" , optimizer=OPTIMIZER, metrics=["mean_squared_error"])
model.fit(x, x, epochs=100)
t=model.predict(x)


dif = numpy.zeros(len(x), dtype=numpy.float)





for i in range(0,len(t)):
    dif[i]=(numpy.linalg.norm(x[i]-t[i]))

print(dif.sum()/len(x))
    




