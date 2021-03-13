import pandas as pd
import numpy as np
from scipy import stats 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# to load model
import pickle

fileName = "DecisionTreeRegressor.sav"
reg = pickle.load(open(fileName, 'rb'))

df = pd.read_csv("training.csv" )

var = df[['util', 'cp', 'bboxArea','bboxAr', 'numPins']]
sc = StandardScaler()
var = sc.fit_transform(var)
label = df['numVias']

# save inference CSV
with open('inference.csv', 'wb') as f:
    f.write(b"numVias\n") 
    np.savetxt(f, reg.predict(var), delimiter=",", fmt='%.3f')

