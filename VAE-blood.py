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

from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

from sklearn.covariance import EmpiricalCovariance, MinCovDet

OPTIMIZER = Adam(lr=0.0001, decay=8e-9)
from sklearn import svm



# Define model
model = Sequential()
model.add(Dense(6, input_dim=8, activation= "relu"))
model.add(Dense(4, activation= "relu"))
model.add(Dense(2, activation= "relu"))
model.add(Dense(4, activation= "relu"))
model.add(Dense(6, activation= "relu"))
model.add(Dense(8, activation= "sigmoid"))


 



filename = 'blood.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
td = list(reader)
data = numpy.array(td).astype('str')


x = data[:, 0:4]  # select columns 1 through end

imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

x = numpy.array(x).astype('float')

w = data[:, 4]   # select column 0, the stock price

w = numpy.array(w).astype('float')

amount_scaler = StandardScaler().fit(x[:])
x[:] = amount_scaler.transform(x[:])


print(len(x))







x2 = numpy.zeros((len(x)*(len(x)),8), dtype=numpy.float)
dif = numpy.zeros((len(x)*(len(x))), dtype=numpy.float)
c=0
for i in range(0,len(x)):
            for p in range(0,len(x)):

                if (i!=p):
                    t1 = np.concatenate((x[i], x[p]), axis=None)

                    x2[c] = t1

                    c=c+1


model.compile(loss= "mean_squared_error" , optimizer=OPTIMIZER, metrics=["mean_squared_error"])
model.fit(x2, x2, epochs=10,verbose=1, validation_split=0.1)
t=model.predict(x2)



for i in range(0,len(t)):
    dif[i]=(numpy.linalg.norm(x2[i]-t[i]))

print(dif.sum()/len(x2))
    




