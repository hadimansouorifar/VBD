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
model.add(Dense(12, input_dim=16, activation= "relu"))
model.add(Dense(8, activation= "relu"))
model.add(Dense(6, activation= "relu"))
model.add(Dense(8, activation= "relu"))
model.add(Dense(12, activation= "relu"))
model.add(Dense(16, activation= "sigmoid"))


 



cancer_data = np.genfromtxt(
 fname ='breast-cancer-wisconsin.data', delimiter= ',', dtype= float)

cancer_data = np.delete(arr = cancer_data, obj= 0, axis = 1)
x = cancer_data[:,range(0,9)]
w = cancer_data[:,9]
w = numpy.array(w).astype('int')
imp = Imputer(missing_values="NaN", strategy='median', axis=0)
x = imp.fit_transform(x)

for i in range(0, len(x)):

    if (w[i] == 4):
        w[i]=1

    else:
        w[i] = 0


amount_scaler = StandardScaler().fit(x[:])
x[:] = amount_scaler.transform(x[:])
print(len(x))



mino=[]
majo=[]
for i in range(0, len(x)):

    if (w[i] == 1):
        mino.append(x[i])
    else:
        majo.append(x[i])


    






x2 = numpy.zeros((len(mino)*(len(mino)),16), dtype=numpy.float)
dif = numpy.zeros((len(majo)*(len(majo))), dtype=numpy.float)
dif2 = numpy.zeros((len(majo)*(len(majo))), dtype=numpy.float)

c=0
for i in range(0,len(mino)):
            for p in range(0,len(mino)):

                if (i!=p):
                    t1 = np.concatenate((mino[i], mino[p]), axis=None)

                    x2[c] = t1

                    c=c+1


model.compile(loss= "mean_squared_error" , optimizer=OPTIMIZER, metrics=["mean_squared_error"])
model.fit(mino, mino, epochs=10,verbose=1, validation_split=0.1)
t=model.predict(majo)



for i in range(0,len(t)):
    dif[i]=(numpy.linalg.norm(majo[i]-t[i]))

print(dif.sum()/len(majo))
    


t=model.predict(mino)



for i in range(0,len(t)):
    dif2[i]=(numpy.linalg.norm(mino[i]-t[i]))

print(dif.sum()/len(mino))



